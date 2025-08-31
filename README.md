# Remesher
Remesher is a Python-based implementation of an algorithm inspired by the paper [Interactive Geometry Remeshing](https://geometry.caltech.edu/pubs/AMD02.pdf) by Alliez et al. 

A geometric processing project that remeshes 3D triangle meshes via 2D harmonic parameterization, area distortion mapping, and dithering-based sampling. This project was developed for the Geometric Modeling course at NYU, and is based on techniques introduced [Interactive Geometry Remeshing](https://geometry.caltech.edu/pubs/AMD02.pdf) by Alliez et al. 


### Overview
This repository implements a remeshing algorithm that:
- Projects a 3D mesh to 2D via harmonic parameterization
- Computes area distortion maps 
- Uses dithering techniques (Bayer matrix or naive thresholding) to sample new mesh points in 2D
- Employs a Delaunay triangulation to obtain new triangles for the mesh, and reconstructs the new mesh in 3D via barycentric projection. 
The core of this functionality is encapsulated in the Remesher class (in `remesher.py`), and demonstrated through Jupyter notebooks (`Remesher_V1.ipynb` to `Remesher_V4.ipynb`).
