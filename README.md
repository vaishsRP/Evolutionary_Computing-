# Evolutionary_Computing-

This project implements an evolutionary algorithm that jointly evolves a robot’s **morphology** (body structure) and **controller** to maximize task performance in the ARIEL simulation environment.

Rather than optimizing control alone, the algorithm searches over coupled body–brain designs, allowing structure and behavior to adapt together through selection and mutation.

---

## Problem Overview

Designing effective robots requires coordinating physical structure with control policies. Hand-designed morphologies constrain performance and generalization. This project frames robot design as a **black-box optimization problem**, solved using evolutionary computing.

The objective is to evolve robot designs that perform well under a fixed evaluation protocol provided by ARIEL.

---

## Method

- **Representation**  
  Each individual encodes:
  - Morphological parameters (robot structure)
  - Controller parameters (movement policy)

- **Evolutionary Algorithm**
  - Population-based search
  - Fitness-based selection
  - Mutation over morphology and control
  - Elitism to preserve high-performing designs

- **Evaluation**
  - Fitness computed via ARIEL simulator
  - Identical evaluation conditions across generations

---

## Files

- `Code_with_plot.py`  
  Main evolutionary algorithm and training loop. Includes logging and performance visualization.

- `Competition_tester.py`  
  Evaluation script used to test evolved robots under competition conditions.

- `Best Robot/`  
  Saved parameters and configuration of the highest-performing evolved robot.

- `Report and Video/`  
  Technical report describing the method and results, along with a short demonstration video.

---

## Results

The evolutionary process consistently discovers robot designs that outperform baseline configurations.  
Performance improves through coordinated changes in morphology and control, highlighting the importance of joint optimization.

Plots in `Code_with_plot.py` show fitness progression across generations.

---

## Key Takeaways

- Joint evolution of body and controller leads to better solutions than optimizing control alone.
- Evolutionary methods are well-suited for high-dimensional, non-differentiable design spaces.
- The learned designs exhibit non-trivial structure–behavior coupling.



## Notes

This repository focuses on algorithmic design and experimental results.  
Environment setup and simulator details are assumed to follow ARIEL documentation.
