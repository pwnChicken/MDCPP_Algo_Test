# Proof-of-Concept MDCPP Algorithm for drone Swarms

## Lightweight Simulator running in pure Python

- Uses `NumPy` and `Matplotlib` libraries to simulate everything in a 2D environment prior to 3D simulation with physics
- Ensures that the Voronoi Partitions are updated when the drone's capacity is updated manually to simulate the battery charge dropping or injecting wind vectors.
- Why? This will be a lot easier to debug as it will run a lot faster than a 3D simulation that requires the drone to fly a path.
