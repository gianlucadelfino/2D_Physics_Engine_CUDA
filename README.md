# SIMULINO 2000 (2D Physics Engine CUDA)
======================

A scalable C++ Physics Engine with CUDA integration.

## Features:
1. Verlet integration engine.
1. Integration with nVidia CUDA run time library (on available devices) for N-body simulation.
1. Scalable and easy to add new simulations.
1. Basic UI, with buttons and labels.
1. Extensive documentation (in Docs/html/index.html).

## Interactive Simulations:
1. Cloth computed on CPU. The cloth can be dragged around clicking and holding the mouse on it.
1. Galaxy (N-body simulation) for both CUDA or CPU. Clicking and holding will add a very massive star in place.


## Instructions:
1. Cloth Simulation
  - click and hold the left mouse button on the cloth and drag it.
![cloth_simulation](/Docs/Simulino_2000_cloth.gif?raw=true)
1. Galaxy Simulation
  - click and hold the left mouse button anywher on screen and a (vey massive) star will be temporarily added there.
  - Use the buttons `MORE\LESS STARS` to increse\decrese the number of stars by 1024
  - Use the button `SWITCH TO CPU\CUDA` to toggle the cpu\CUDA integration (CUDA integration is available only on nVidia cards)
![cloth_simulation](/Docs/Simulino_2000_galaxy.gif?raw=true)

## Link to win32 builds:
1. https://dl.dropboxusercontent.com/u/3272164/Simulino2000_09112013.zip


# Developer ReadMe

## Instructions to compile:
1. Install CUDA Toolkit (available upon free registration on https://developer.nvidia.com/cuda-toolkit).
1. Clone repository.
1. Open "2D_Phys_Engine_CUDA.vcxproj" with Visual Studio 2012.

## Class Diagram (editable version in `Docs\` folder)
- (click to magnify)
![Class Diagram](/Docs/class_diagram.jpg?raw=true)


