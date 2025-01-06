# Genetic-Algorithms-Image-Processing
This Python program processes images for genetic algorithm optimization. It includes functionalities for loading images, simplifying their color palette, calculating search space sizes, plotting data, and running a genetic algorithm to optimize image transformations based on simplified color schemes.

**Must update code, repo structure and readme**

### Overview
This program is designed for image processing and optimization using genetic algorithms. It offers a variety of functions to load and preprocess images, simplify their color schemes, and calculate the complexity of their color combinations. It also features an extensive genetic algorithm implementation to evolve images towards a target configuration, optimizing their color representation and structure.

### Installation
Ensure Python is installed along with the numpy, matplotlib, and PIL libraries, necessary for handling arrays, plotting, and image processing, respectively:
pip install numpy matplotlib Pillow

### Image Loading and Processing:
`load_image(file, size)`: Loads an image, resizes it, and checks if it's in PNG format, returning the image as a NumPy array and a format flag.
`simplify_colors(image)`: Simplifies an image's color scheme to binary colors based on a threshold, making it easier to analyze and manipulate.

Search Space Calculation:
`compute_search_space_size(file, size)`: Computes the complexity of color combinations for both original and simplified images, providing insight into the processing challenge.

Visualization:
plot_search_space_size(...): Plots the calculated search space sizes for visual comparison between different image types and simplification levels.

Genetic Algorithm for Image Optimization:
Functions for generating populations, selecting fittest individuals, performing crossover and mutation, and replacing populations ensure robust genetic algorithm operations tailored for image optimization.
genetic_algorithm(...): Orchestrates the genetic algorithm operations over multiple iterations to optimize an image towards a predefined goal.

Fitness and Mutation Functions:
Include various ways to evaluate the fitness of images compared to a target and mutate images to introduce variability and explore new solutions.

Usage
Run the program to process images, compute their search spaces, and optimize them using genetic algorithms. Outputs include processed images, search space calculations, and visualizations of algorithmic performance.

Example Code
Here's how to run a genetic algorithm on a sample image:
```python
file_path = 'path/to/image.png'
n_iterations = 100
population_size = 50
crossover_rate = 0.8
mutation_rate = 0.01
replacement_rate = 1.0

best_image, best_score = genetic_algorithm(
    file_path, n_iterations, population_size, crossover_rate, mutation_rate, replacement_rate
)

print(f"Best score: {best_score}")
plt.imshow(best_image)
plt.title('Optimized Image')
plt.show()
```

This program provides a comprehensive toolset for manipulating and optimizing images using genetic algorithms. It can be particularly useful in fields such as digital art, pattern recognition, and other areas where image processing and optimization are required.
