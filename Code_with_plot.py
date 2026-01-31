"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import copy, sys, traceback

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import json
from deap import base, creator, tools
import copy, random
# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (construct_mjspec_from_graph)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (HighProbabilityDecoder,save_graph_as_json)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
import networkx as nx
from datetime import datetime
import time


# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 50 #42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 20
TARGET_POSITION = [5, 0, 0.5]

NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)

def log_with_timestamp(message, level="INFO"):
    """Print message with timestamp and formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == "HEADER":
        print(f"\n{'='*80}")
        print(f"[{timestamp}] {message}")
        print(f"{'='*80}")
    elif level == "BODY":
        print(f"\n{'─'*80}")
        print(f"[{timestamp}] 🤖 {message}")
        print(f"{'─'*80}")
    elif level == "BRAIN":
        print(f"[{timestamp}] 🧠 {message}")
    elif level == "CHECKPOINT":
        print(f"[{timestamp}] ✓ {message}")
    elif level == "WARNING":
        print(f"[{timestamp}] ⚠️  {message}")
    else:
        print(f"[{timestamp}] {message}")

def dict_to_graph(graph_dict):
    """
    Convert JSON/dict representation of robot graph to a NetworkX DiGraph.
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_dict["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, type=node["type"], rotation=node["rotation"])
    
    # Add edges
    for edge in graph_dict["edges"]:
        source = edge["source"]
        target = edge["target"]
        G.add_edge(source, target, face=edge["face"])
    
    return G

def import_json(filename: str = "robot_graph.json", directory: Path = DATA) -> dict:

    file_path = directory / filename
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def import_best_individual(directory: Path = DATA / "best_saved" / "best_individual") ->  dict:
    """
    Loads the best individual saved in 'save_best_body':
      - robot_graph.json
      - genotype.json
      - best_brain.npy
      - brain_meta.json
    Returns a dict with keys:
      'robot_graph', 'genotype', 'best_brain', 'brain_meta'
    """
    directory = Path(directory)
    print(DATA)
    # 1) Load robot graph
    with open(directory / "robot_graph.json", "r") as f:
        robot_graph = json.load(f)

    # 2) Load genotype
    with open(directory / "genotype.json", "r") as f:
        genotype = json.load(f)

    # 3) Load brain vector
    best_brain = np.load(directory / "best_brain.npy", allow_pickle=True)

    # 4) Load brain meta
    with open(directory / "brain_meta.json", "r") as f:
        brain_meta = json.load(f)

    return {"robot_graph": robot_graph,"genotype": genotype,"best_brain": best_brain,"brain_meta": brain_meta}

def visualize_best(duration: int = 60, mode: str = "viewer", spawn_pos=[5, 0, 0.5]):    
    best_ind = import_best_individual()
    best_brain = best_ind["best_brain"]
    brain_meta = best_ind["brain_meta"]
    best_genotype = best_ind["genotype"]
    robot_graph = best_ind["robot_graph"]
    robot_graph = dict_to_graph(robot_graph)
    print(brain_meta["input_size"], brain_meta["hidden_size"], brain_meta["output_size"])


    # # --- Reconstruct MuJoCo model ---
    core = construct_mjspec_from_graph(robot_graph)      
    world = OlympicArena()
    world.spawn(core.spec, position=spawn_pos)  
    model = world.spec.compile()  # compile using the world's internal spec
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # --- Prepare controller ---
    input_size = len(data.qpos)  # automatically matches model
    hidden_size = best_ind["brain_meta"]["hidden_size"]
    output_size = model.nu

    print(input_size, hidden_size, output_size)

    genome = np.array(best_brain, dtype=np.float32).ravel()
    idx = 0
    w1 = genome[idx: idx + input_size*hidden_size].reshape((input_size, hidden_size)); idx += input_size*hidden_size
    w2 = genome[idx: idx + hidden_size*hidden_size].reshape((hidden_size, hidden_size)); idx += hidden_size*hidden_size
    w3 = genome[idx: idx + hidden_size*output_size].reshape((hidden_size, output_size))

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=test_persistent_controller(w1, w2, w3), tracker=tracker)
    # ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # --- Run simulation ---
    if mode == "simple":
        simple_runner(model, data, duration=duration)
        show_xpos_history(tracker.history["xpos"][0])
        print(fitness_function(tracker.history["xpos"][0]))
    elif mode == "viewer":
        viewer.launch(model=model, data=data)
        show_xpos_history(tracker.history["xpos"][0])
        print(fitness_function(tracker.history["xpos"][0]))
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'viewer' or 'simple'.")


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance

def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model,data,save_path=save_path,save=True)

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()


def nn_controller(model: mj.MjModel,data: mj.MjData) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))


    # Scale the outputs
    return outputs * np.pi

def test_persistent_controller(w1, w2, w3):

    def test_controller(model: mj.MjModel,data: mj.MjData) -> npt.NDArray[np.float64]:

        # forward pass
        inputs = np.array(data.qpos, dtype=np.float32)  # shape (input_size,)
        layer1 = np.tanh(np.dot(inputs, w1))
        layer2 = np.tanh(np.dot(layer1, w2))
        # outputs = np.tanh(np.dot(layer2, w3))
        outputs = np.sin(np.dot(layer2, w3))

        return outputs * np.pi

    return test_controller

def experiment(robot: Any,controller: Controller,duration: int = 15,mode: ViewerTypes = "viewer") -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    args: list[Any] = [] 
    kwargs: dict[Any, Any] = {} 

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(model,data,duration=duration)
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(model,data,duration=duration,video_recorder=video_recorder)
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(model=model,data=data)
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(model=model,data=data)
    # ==================================================================== #

def test_experiment(robot: Any, duration: int = 20, mode: ViewerTypes = "viewer", 
                    body_id: str = "unknown") -> None:
    """Modified with progress logging"""
    
    log_with_timestamp(f"Starting brain evolution for {body_id}", "BRAIN")
    start_time = time.time()
    
    NUM_GENERATIONS_BRAIN = 50
    BRAIN_POP_SIZE = 40
    ELITES = 1
    BEST_FOR_NOW = -np.inf
    PATIENCE = 7

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    toolbox = deap_setup()

    mj.mj_resetData(model, data)

    input_size = len(data.qpos)
    hidden_size = 32
    output_size = model.nu

    n1 = input_size * hidden_size
    n2 = hidden_size * hidden_size
    n3 = hidden_size * output_size
    brain_genome_length = n1 + n2 + n3
    
    brains_population = create_brain_population(brain_genome_length, BRAIN_POP_SIZE)

    best_fitness_history = []
    best_individual_ever = tools.HallOfFame(1)
    no_improvement_counter = 0

    for gen in range(NUM_GENERATIONS_BRAIN):
        gen_start = time.time()
        
        # Evaluate existing population
        log_with_timestamp(f"  Brain Gen {gen}/{NUM_GENERATIONS_BRAIN} - Evaluating {len(brains_population)} brains...", "INFO")
        
        for i, brain_ind in enumerate(brains_population):
            if i % 10 == 0:  # Progress indicator every 10 brains
                print(f"    -> Evaluating brain {i}/{len(brains_population)}...", end='\r')
            
            evaluate_brain_on_body(
                model=model, data=data, world=world, brain_ind=brain_ind,
                n1=n1, n2=n2, n3=n3,
                input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                duration=duration, brain_id=f"{body_id}_gen{gen}_brain{i}"
            )

        # Evolution
        new_brains = evolution_brain(brains_population)
        invalid = [ind for ind in new_brains if not hasattr(ind, "fitness") or not ind.fitness.valid]

        if invalid:
            log_with_timestamp(f"  Brain Gen {gen}/{NUM_GENERATIONS_BRAIN} - Evaluating {len(invalid)} new brains...", "INFO")
            
            for i, new_brain_ind in enumerate(invalid):
                if i % 10 == 0:
                    print(f"    -> Evaluating new brain {i}/{len(invalid)}...", end='\r')
                
                evaluate_brain_on_body(
                    model=model, data=data, world=world, brain_ind=new_brain_ind,
                    n1=n1, n2=n2, n3=n3,
                    input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                    duration=duration, brain_id=f"{body_id}_gen{gen}_newbrain{i}"
                )
        
        new_brains_population = invalid
        
        # Selection
        total_population = brains_population + new_brains_population
        elites = tools.selBest(total_population, ELITES)
        survivors = toolbox.select_brain(total_population, BRAIN_POP_SIZE-ELITES)
        brains_population[:] = survivors + elites 

        best = tools.selBest(brains_population, 1)[0]
        best_fitness_history.append(best.fitness.values[0])
        best_individual_ever.update(brains_population)

        # Check improvement
        if best.fitness.values[0] > BEST_FOR_NOW:
            BEST_FOR_NOW = best.fitness.values[0]
            no_improvement_counter = 0
            log_with_timestamp(f"  Brain Gen {gen}: NEW BEST fitness = {BEST_FOR_NOW:.4f} ⬆️", "CHECKPOINT")
        else:
            no_improvement_counter += 1
            log_with_timestamp(f"  Brain Gen {gen}: fitness = {best.fitness.values[0]:.4f} (no improvement x{no_improvement_counter})", "INFO")

        gen_time = time.time() - gen_start
        log_with_timestamp(f"  Brain Gen {gen} completed in {gen_time:.1f}s", "INFO")

        # Early stopping
        if no_improvement_counter >= PATIENCE:
            log_with_timestamp(f"Early stopping after {gen} generations (no improvement for {PATIENCE} gens)", "WARNING")
            break
    
    total_time = time.time() - start_time
    best_brain = best_individual_ever[0]
    best_brain_meta = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    
    log_with_timestamp(f"Brain evolution for {body_id} completed in {total_time:.1f}s. Final fitness: {best_brain.fitness.values[0]:.4f}", "CHECKPOINT")
    
    return best_brain, best_brain_meta, (best_brain.fitness.values[0],)

def evaluate_brain_on_body(
    model: mj.MjModel,
    data: mj.MjData,
    world: OlympicArena,
    brain_ind,
    n1: int, n2: int, n3: int,
    input_size: int, hidden_size: int, output_size: int,
    duration: int = 15,
    brain_id: str = "unknown"):
    """Modified with detailed logging"""
    
    eval_start = time.time()
    
    toolbox = deap_setup()
    genome = np.asarray(brain_ind, dtype=np.float32).ravel().copy()
    idx = 0
    w1 = genome[idx: idx + n1].reshape((input_size, hidden_size)); idx += n1
    w2 = genome[idx: idx + n2].reshape((hidden_size, hidden_size)); idx += n2
    w3 = genome[idx: idx + n3].reshape((hidden_size, output_size))

    def clear_tracker_history(tracker):
        n_tracked = len(getattr(tracker, "to_track", {}))
        if n_tracked <= 0:
            n_tracked = 1
        for attr in list(tracker.history.keys()):
            tracker.history[attr] = {i: [] for i in range(n_tracked)}

    def xs_from_history(tracker):
        xpos_data = tracker.history.get("xpos", {})
        if 0 not in xpos_data or not xpos_data[0]:
            return []
        return [pos[0] for pos in xpos_data[0]]

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=test_persistent_controller(w1, w2, w3), tracker=tracker)
    
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    args: list[Any] = []
    kwargs: dict[Any, Any] = {}
    mj.set_mjcb_control(lambda m, d, ctrl=ctrl: ctrl.set_control(m, d, *args, **kwargs))

    # Checkpoint configuration
    checkpoints = [-0.2, 0.0, 0.2, 0.5, 1.5, 2.5, 3.0]
    duration_increment = [5, 5, 5, 45, 20, 45, 60]
    max_reruns = 8
    max_total_duration = 200

    def highest_crossed_index(max_x_val):
        hi = -1
        for i, cp in enumerate(checkpoints):
            if max_x_val >= cp:
                hi = i
        return hi

    # Initial run
    current_duration = int(duration)
    mj.mj_resetData(model, data)
    
    try:
        simple_runner(model, data, duration=current_duration)
    except Exception as e:
        log_with_timestamp(f"ERROR in initial sim for {brain_id}: {e}", "WARNING")
        brain_ind.fitness.values = (-float('inf'),)
        mj.set_mjcb_control(None)
        return

    x_positions = xs_from_history(tracker)
    max_x = max(x_positions) if x_positions else float("-inf")
    last_hi = highest_crossed_index(max_x)
    
    reruns = 0
    rerun_log = []  # Track all reruns

    # Rerun loop with detailed logging
    while True:
        # Safety checks
        if last_hi < 0:
            break
        
        if last_hi >= len(checkpoints) - 1:
            log_with_timestamp(f"      {brain_id}: Reached highest checkpoint!", "CHECKPOINT")
            break
        
        if reruns >= max_reruns:
            log_with_timestamp(f"      {brain_id}: Max reruns ({max_reruns}) reached", "WARNING")
            break
        
        added = duration_increment[last_hi]
        if current_duration + added > max_total_duration:
            log_with_timestamp(f"      {brain_id}: Duration cap ({max_total_duration}s) reached", "WARNING")
            break
        
        # Perform rerun
        current_duration += added
        reruns += 1
        
        rerun_info = f"cp{last_hi}({checkpoints[last_hi]:.1f})->+{added}s"
        rerun_log.append(rerun_info)
        
        # Only print detailed info for significant checkpoints
        if checkpoints[last_hi] >= 0.9:
            print(f"      {brain_id}: Rerun #{reruns}: {rerun_info}, total_dur={current_duration}s")

        clear_tracker_history(tracker)
        mj.mj_resetData(model, data)
        
        try:
            simple_runner(model, data, duration=current_duration)
        except Exception as e:
            log_with_timestamp(f"      ERROR in rerun #{reruns} for {brain_id}: {e}", "WARNING")
            break

        xs = xs_from_history(tracker)
        new_max_x = max(xs) if xs else float("-inf")
        new_hi = highest_crossed_index(new_max_x)

        # Check progress
        if new_hi <= last_hi:
            max_x = new_max_x
            break

        # Progress made
        max_x = new_max_x
        last_hi = new_hi

    # Calculate fitness
    final_positions = tracker.history.get("xpos", {})
    final_positions = final_positions.get(0, []) if final_positions else []
    
    if final_positions:
        brain_ind.fitness.values = (toolbox.evaluate_fitness(final_positions),)
        fitness_val = brain_ind.fitness.values[0]
    else:
        log_with_timestamp(f"      {brain_id}: No position data!", "WARNING")
        brain_ind.fitness.values = (-float('inf'),)
        fitness_val = -float('inf')

    # Log summary if there were reruns
    eval_time = time.time() - eval_start
    if reruns > 0 and max_x >= 1.1:  # Only log if reached checkpoint 0.5 or higher
        print(f"      {brain_id}: {reruns} reruns, {eval_time:.1f}s, fitness={fitness_val:.4f}, path: {' -> '.join(rerun_log)}")

    # Cleanup
    clear_tracker_history(tracker)
    mj.set_mjcb_control(None)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax) 

def deap_setup():
    toolbox = base.Toolbox()
    toolbox.register("evaluate_fitness", fitness_function)
    toolbox.register("mate", tools.cxUniform, indpb = 0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.6)
    toolbox.register("select_body", tools.selTournament, tournsize=4)
    toolbox.register("select_brain", tools.selTournament, tournsize=10)

    return toolbox

def create_body_population(genotype_size: int =64, pop_size: int = 2):
    population_genotype = []

    for i in range(pop_size):

        type_p_genes = RNG.random(genotype_size).astype(np.float32)
        conn_p_genes = RNG.random(genotype_size).astype(np.float32)
        rot_p_genes = RNG.random(genotype_size).astype(np.float32)
        genotype = [type_p_genes,conn_p_genes,rot_p_genes]
        ind = creator.Individual(genotype)
        population_genotype.append(ind)

        
    return population_genotype

def random_move(model, data, to_track, HISTORY) -> None:
    num_joints = model.nu
    hinge_range = np.pi
    rand_moves = np.random.uniform(
        low=-hinge_range,  # -pi/2
        high=hinge_range,  # pi/2
        size=num_joints,
    )

    delta = 0.1
    data.ctrl += rand_moves * delta
    data.ctrl = np.clip(data.ctrl, -np.pi, np.pi)
    HISTORY.append(to_track[0].xpos.copy())

def create_body_population_with_random_move_selection(genotype_size: int =64, pop_size: int = 2):
    population_genotype = []

    while len(population_genotype) < pop_size:
        type_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32) 
        conn_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32) 
        rot_p_genes = RNG.uniform(-100, 100, genotype_size).astype(np.float32) 
        genotype = [type_p_genes,conn_p_genes,rot_p_genes]
        ind = creator.Individual(genotype)
        core, robot_graph =initialize_NDE(ind)

        mj.set_mjcb_control(None)  # DO NOT REMOVE
        world = OlympicArena()
        world.spawn(core.spec, position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        toolbox = deap_setup()

        geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        HISTORY=[]
        mj.set_mjcb_control(lambda m, d: random_move(m, d, to_track, HISTORY))
        # give each individual 3 chances to exit circle in order to keep possible good bodies that just got unlucky
        for i in range(2):
            simple_runner(model,data,duration=10)
            # PUT CIRCLE CENTER NOT AS SPAWN POS BECAUSE ROBOT SOMETIMES CAN FALL FROM SKY AND ROLL THUS EXITING CIRCLE
            length = int(len(HISTORY)/2)
            x_start, y_start, z_start = HISTORY[length]
            x_end, y_end, z_end = HISTORY[-1]
            
            # Define circle center and radius
            circle_center = np.array([x_start, y_start])  # x, y
            circle_radius = 0.2  # adjust as needed
            # Get last position (x, y)
            last_pos = np.array([x_end, y_end])
            # Check if outside the circle
            dist = np.linalg.norm(last_pos - circle_center)
            if dist > circle_radius:
                population_genotype.append(ind)
                break
            mj.mj_resetData(model, data)
            mj.set_mjcb_control(None)
    print("Finished creating body population.")
    return population_genotype

def create_brain_population(genome_length_brain,  brain_pop_size: int=10):
    toolbox = base.Toolbox()
    toolbox.register("attr_param", lambda: random.uniform(-1, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_param, n=genome_length_brain)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox.population(n=brain_pop_size)

  
def evolution_body(population):

    toolbox = deap_setup()
    parents = toolbox.select_body(population, len(population))
    parents = list(map(copy.deepcopy, parents))
    random.shuffle(parents)
    new_children =[]
    for parent1 , parent2 in zip(parents[::2], parents[1::2]):
        
        child1, child2 = parent1, parent2
        for i in range(3):
            if random.random() < 1:
                toolbox.mate(child1[i], child2[i])
                del child1.fitness.values
                del child2.fitness.values
  
            if random.random() < 0.6:
                toolbox.mutate(child1[i])
                toolbox.mutate(child2[i])
                del child1.fitness.values
                del child2.fitness.values
        new_children.extend([child1, child2])
    return new_children

def evolution_brain(population):

    toolbox = deap_setup()
    parents = toolbox.select_brain(population, len(population))
    parents = list(map(copy.deepcopy, parents))
    random.shuffle(parents)
    new_children =[]
    for parent1 , parent2 in zip(parents[::2], parents[1::2]):
        
        child1, child2 = parent1, parent2
        if random.random() < 1:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        if random.random() < 0.6:
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values
            del child2.fitness.values
        new_children.extend([child1, child2])
    return new_children

def save_best_body(robot_graph, individual, out_dir: Path = DATA / "best_saved", name: str = "best"):
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) save robot graph (uses your existing helper)
    save_graph_as_json(robot_graph, out_dir / "robot_graph.json")

    # 2) save genotype (convert numpy arrays to lists)
    genotype_serializable = []
    for arr in individual:  # individual is [type_p_genes, conn_p_genes, rot_p_genes]
        if hasattr(arr, "tolist"):
            genotype_serializable.append(arr.tolist())
        else:
            genotype_serializable.append(list(arr))
    with open(out_dir / "genotype.json", "w") as f:
        json.dump(genotype_serializable, f, indent=2)

    # 3) save brain (if available) and meta
    if hasattr(individual, "best_brain"):
        np.save(out_dir / "best_brain.npy", np.asarray(individual.best_brain, dtype=np.float32))
    if hasattr(individual, "brain_dimensions"):
        with open(out_dir / "brain_meta.json", "w") as f:
            json.dump(individual.brain_dimensions, f, indent=2)

    print(f"Saved best body assets to {out_dir}")


def initialize_NDE(ind):

    p_matrices = NDE.forward(ind)
    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)    
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(p_matrices[0],p_matrices[1],p_matrices[2])
    core = construct_mjspec_from_graph(robot_graph)
    
    return core, robot_graph

def read_fitness_file(filename):
    """
    Reads the fitness data file into a numpy array.
    Expected format: generation, fitness_1, fitness_2, std_dev
    """
    data = np.loadtxt(filename, delimiter=',')
    return data

def extract_fitness_data(data, use_column=1):
    """
    use_column: 1 for fitness_1, 2 for fitness_2
    """
    generations = data[:, 0]
    fitness = data[:, use_column]
    std_dev = data[:, 3]
    return generations, fitness, std_dev

def plot_fitness_with_std(generations, fitness, std_dev, algorithm="Algorithm", window=5, save_path="fitness_plot.png"):
    """
    Plots fitness (raw + moving average) and standard deviation.
    - generations: x-axis values
    - fitness: y-axis values
    - std_dev: per-generation std deviation
    """
    
    # Compute moving average
    if window > 1 and len(fitness) >= window:
        ma_fitness = np.convolve(fitness, np.ones(window)/window, mode='valid')
        ma_generations = generations[window-1:]
    else:
        ma_fitness = fitness
        ma_generations = generations

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness, 'b-', label='Fitness', linewidth=2)
    
    # Plot std deviation as shaded region
    plt.fill_between(generations, fitness - std_dev, fitness + std_dev, color='gray', alpha=0.3, label='± Std Dev(Across all bodies)')
    
    # Labels and aesthetics
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title(f"Fitness over Generations ({algorithm})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    log_with_timestamp("STARTING EVOLUTIONARY RUN", "HEADER")
    overall_start = time.time()
    
    POP_SIZE_BODY = 10
    NUM_GENERATIONS_BODY = 50
    ELITES = 1
    
    log_with_timestamp("Creating initial body population...", "INFO")
    population = create_body_population_with_random_move_selection(pop_size=POP_SIZE_BODY)
    log_with_timestamp(f"Initial population created: {len(population)} bodies", "CHECKPOINT")
    
    toolbox = deap_setup()
    best_fitness_history = []

    best_robot_graph = None
    best_individual_snapshot = None
    best_fitness_value = -float("inf")

    for gen in range(NUM_GENERATIONS_BODY):
        log_with_timestamp(f"GENERATION {gen}/{NUM_GENERATIONS_BODY}", "HEADER")
        gen_start_time = time.time()
        
        # Evaluate population
        log_with_timestamp(f"Evaluating {len(population)} bodies in population...", "INFO")

        fitness_avg_runner = 0.0
        fitness_history = []        
        for idx, ind in enumerate(population):
            body_id = f"gen{gen}_body{idx}"
            log_with_timestamp(f"Body {idx+1}/{len(population)}: Starting evaluation...", "BODY")
            body_start = time.time()

            core, robot_graph = initialize_NDE(ind)
            best_brain_ind, brain_dimensions_ind, fitness_ind = test_experiment(
                robot=core, mode="simple", body_id=body_id
            )
            
            fitness_avg_runner+= fitness_ind[0]
            fitness_history.append(fitness_ind[0])

            ind.fitness.values = fitness_ind
            ind.brain_dimensions = copy.deepcopy(brain_dimensions_ind)
            ind.best_brain = np.asarray(best_brain_ind, dtype=np.float32).ravel().copy()

            # Save individual
            indiv_dir = DATA / "all_individuals" / f"gen_{gen}_ind_{idx}"
            save_best_body(robot_graph, ind, out_dir=indiv_dir, name=f"ind_{gen}_{idx}")

            # Check if best
            fitness_value = float(fitness_ind[0])
            body_time = time.time() - body_start
            
            log_with_timestamp(
                f"Body {idx+1}/{len(population)} completed in {body_time:.1f}s. Fitness: {fitness_value:.4f}",
                "CHECKPOINT"
            )
            
            if fitness_value > best_fitness_value:
                best_fitness_value = fitness_value
                best_robot_graph = robot_graph
                best_individual_snapshot = copy.deepcopy(ind)
                log_with_timestamp(
                    f"🎉 NEW GLOBAL BEST! Fitness: {best_fitness_value:.4f} (gen {gen}, body {idx})",
                    "CHECKPOINT"
                )
                save_best_body(best_robot_graph, best_individual_snapshot, 
                             out_dir=DATA / "best_saved", name="best_individual")

        # Evolution
        log_with_timestamp(f"Creating offspring via evolution...", "INFO")
        new_population = evolution_body(population)
        invalid = [i for i in new_population if not hasattr(i, "fitness") or not i.fitness.valid]
        new_population = invalid
        
        log_with_timestamp(f"Evaluating {len(new_population)} new offspring...", "INFO")


        for idx, new_ind in enumerate(new_population):
            body_id = f"gen{gen}_newbody{idx}"
            log_with_timestamp(f"New Body {idx+1}/{len(new_population)}: Starting evaluation...", "BODY")
            body_start = time.time()

            core, robot_graph = initialize_NDE(new_ind)
            best_brain_new_ind, brain_dimensions_new_ind, fitness_new_ind = test_experiment(
                robot=core, mode="simple", body_id=body_id
            )

            new_ind.fitness.values = fitness_new_ind
            new_ind.best_brain = np.asarray(best_brain_new_ind, dtype=np.float32).ravel().copy()
            new_ind.brain_dimensions = copy.deepcopy(brain_dimensions_new_ind)

            # Save individual
            indiv_dir = DATA / "all_individuals" / f"gen_{gen}_newind_{idx}"
            save_best_body(robot_graph, new_ind, out_dir=indiv_dir, name=f"newind_{gen}_{idx}")

            fval = float(fitness_new_ind[0])
            body_time = time.time() - body_start
            
            log_with_timestamp(
                f"New Body {idx+1}/{len(new_population)} completed in {body_time:.1f}s. Fitness: {fval:.4f}",
                "CHECKPOINT"
            )
            
            if fval > best_fitness_value:
                best_fitness_value = fval
                best_robot_graph = robot_graph
                best_individual_snapshot = copy.deepcopy(new_ind)
                log_with_timestamp(
                    f"🎉 NEW GLOBAL BEST from offspring! Fitness: {best_fitness_value:.4f}",
                    "CHECKPOINT"
                )
                save_best_body(best_robot_graph, best_individual_snapshot,
                             out_dir=DATA / "best_saved", name="best_individual")

        fitness_avg_runner = fitness_avg_runner/len(new_population)
        # Selection
        total_population = population + new_population
        elites = tools.selBest(total_population, ELITES)
        survivors = toolbox.select_body(total_population, POP_SIZE_BODY-ELITES)
        population[:] = survivors + elites 
        
        best = tools.selBest(population, 1)[0]
        best_fitness_history.append(best.fitness.values[0])
        
        gen_time = time.time() - gen_start_time
        log_with_timestamp(
            f"Generation {gen} completed in {gen_time:.1f}s. Best in gen: {best.fitness.values[0]:.4f}",
            "CHECKPOINT"
        )
        gen_fitness_values = []
        gen_fitness_values = [gen, best.fitness.values[0], fitness_avg_runner, np.std(np.array(fitness_history))]

        
        with open('fitness_graph.txt', 'a') as f:
            line = ", ".join(map(str, gen_fitness_values))
            f.write(line + "\n")

    # Final summary
    total_time = time.time() - overall_start
    log_with_timestamp("EVOLUTION COMPLETED!", "HEADER")
    log_with_timestamp(f"Total time: {total_time/60:.1f} minutes", "INFO")
    log_with_timestamp(f"Best fitness achieved: {best_fitness_value:.4f}", "INFO")
    
    if best_robot_graph is not None and best_individual_snapshot is not None:
        save_best_body(best_robot_graph, best_individual_snapshot,
                      out_dir=DATA / "best_saved", name="best_individual")
    
    print(f"\nFinal best brain dimensions: {best_individual_snapshot.brain_dimensions}")
    print(f"Final best fitness: {best_individual_snapshot.fitness.values[0]}")



# # # --------------- Uncomment to visualise best individual ---------------
visualize_best(duration=30, mode="viewer", spawn_pos=SPAWN_POS)



# # # --------------- Uncomment for Plots ---------------
# data = read_fitness_file("fitness_graph.txt")

# # # --- Choose which fitness column to use (1 for best fitness graph, 2 for average fitness graph) ---
# generations, fitness, std_dev = extract_fitness_data(data, use_column=2)  # or 1

# plot_fitness_with_std(
#     generations,
#     fitness,
#     std_dev,
#     algorithm="Evolution of Body",
#     window=3,
#     save_path="fitness_plot_final.png"
# )


# # # --------------- Uncomment to run the evolution code ---------------
# if __name__ == "__main__":
#     main()
