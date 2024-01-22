# Particle filter
<p align="center">
<img src="https://github.com/maxipalay/particle_filter/assets/41023326/97676c99-0d67-4e02-a886-2d8d1d1b1bc0">
</p>

Python implementation of a Particle Filter for robot localization. This work was done for <em>Machine Learning and Artificial Intelligence for Robotics</em>, an elective course I took in my MSR journey. The implementation is based on the particle filter as explained in <em>Probabilistic Robotics, by Thrun, Sebastian; Wolfram, Burgard; Fox, Dieter. MIT Press Books, 2006</em>.

This is a modified version of the one requested for the aforementioned course.

# Files
- `main.py` - the executable
- `data_utils.py` - utilities for dealing with the dataset
- `measurement_model.py` - implementation of measurement model and helper functions
- `motion_model.py` - implementation of motion model for diff drive robot
- `particle_filter.py` - class that represents a particle filter
- `plotting.py` - utilities for plotting
- `prob_utils.py` - probability function helpers
- `state.py` - pseudo-enum that represents a state

# Usage
The user should run `main.py` to run the particle filter on the provided data. This provides an estimation of location in a map, while also plotting ground truth data.

At the top of `main` some variables allow to tinker with parameters easily.

# Attribution
The dataset used is a modified version of the work made available by Keith Leung, Yoni Halpern, Tim Barfoot, and Hugh Liu [\*]. It was provided by Prof. Brenna Argall in the Fall 2023 edition of the course <em>Machine Learning and Artificial Intelligence for Robotics</em> at Northwestern University.
<br>
<br>
<em>[*] Leung K Y K, Halpern Y, Barfoot T D, and Liu H H T. “The UTIAS Multi-Robot Cooperative Localization and Mapping Dataset”. International Journal of Robotics Research, 30(8):969–974, July 2011.</em>

# Notes
This work was made in an academic environment, with certain restrictions for pedagogic purposes.

This code is not guaranteed to be free of bugs.
