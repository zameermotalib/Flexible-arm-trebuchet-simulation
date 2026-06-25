# Flexible-arm-trebuchet-simulation
Numerical simulation of a trebuchet with a flexible throwing arm, modelling 3-DOF rigid-flexible body dynamics, energy conservation, and convergence behaviour.
# Flexible-Arm Trebuchet Simulation

A numerical simulation of a trebuchet with a flexible throwing arm, built for a 
university dynamics assignment. Models the system as a 3-degree-of-freedom 
rigid-flexible body problem and solves the equations of motion numerically.

## What it does
- Derives and solves the coupled equations of motion for beam rotation, arm 
  flexibility, and sling/projectile swing
- Uses a soft-contact penalty method to model the projectile leaving the ground
- Detects the release point automatically based on launch angle and direction
- Verifies the solution through energy conservation checks
- Runs a timestep convergence study and a spring-stiffness sensitivity study
- Animates the full launch sequence and plots key results (velocity, energy 
  distribution, energy error)

## Tools
Python, NumPy, Matplotlib (animation + plotting)

## What I learned
Working through this reinforced how sensitive numerical integration schemes 
are to timestep size and how important it is to verify a simulation isn't 
just "running" but actually conserving energy. Building in automatic release 
detection and convergence/sensitivity studies also pushed me to think about 
robustness, not just getting one case to work.
