# 2D Physics Engine CUDA
======================

## NB: This project is currently being refactored and being modernized

A scalable C++ Physics Engine with CUDA integration.

## Features:
1. Verlet integration engine.
1. Integration with Nvidia CUDA run time library (on available devices) for N-body simulation.
1. Scalable architecture and easy to add new simulations.
1. Basic UI, with buttons and labels.
1. Asynchronous multi-threaded level loading.
1. Extensive documentation (in Docs/html/index.html).

## Interactive Simulations:
1. Cloth (computed on CPU). The cloth can be dragged around clicking and holding the mouse on it.
1. Galaxy (N-body simulation) (computed either on CUDA GPU or CPU). Clicking and holding will add a very massive star in place.


## Usage:
1. Cloth Simulation
  - click and hold the left mouse button on the cloth and drag it.

  ![cloth_simulation](/Docs/Simulino_2000_cloth.gif?raw=true)

1. Galaxy Simulation
  - click and hold the left mouse button anywhere on screen and a (very massive) star will be temporarily added there.
  - Use the buttons `MORE\LESS STARS` to increase\decrease the number of stars (by 1024 stars for CUDA or 128 on CPU)
  - Use the button `SWITCH TO CPU\CUDA` to toggle the cpu\CUDA integration
    (CUDA integration is available only if an Nvidia device is present)

  ![cloth_simulation](/Docs/Simulino_2000_galaxy.gif?raw=true)

## How to build:
The project can be built either on linux or windows.

1. Install CUDA Toolkit (available upon free registration on https://developer.nvidia.com/cuda-toolkit).
2. Install Conan Package Manager

```
    pip install conan
```

3. Build with cmake

```
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --parallel
```

Then the program can be launched from the "app" directory

```
    cd build
    ./app/engine
```

## Class Diagram (editable version in `Docs\` folder)
- (click to magnify)
![Class Diagram](/Docs/class_diagram.jpg?raw=true)


License
=======

This Program is released under the MIT license:

* http://www.opensource.org/licenses/MIT

