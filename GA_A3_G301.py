from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random

"""
Authors: Carolina Kogan, Yago Granada, Emma Devys and Rodrigo Sastre
"""
# Load Image
def load_image(file, size):
    """
    Loads an image from a specified file, scales it to a target size, and converts it to a NumPy array.
    The function also checks if the image is in PNG format.

    Args:
    - file (str): The path to the image file to be loaded.
    - size (int): The maximum dimension (width or height) to which the image should be scaled.

    Returns:
    - tuple: A tuple containing the NumPy array of the scaled image and a boolean indicating whether
             the image is in PNG format (True) or not (False).
    """
    img = Image.open(file)
    png = img.format == "PNG"
    factor = max(img.size) / size
    new_width = max(int(img.size[0] / factor), 1)
    new_height = max(int(img.size[1] / factor), 1)
    img = img.resize((new_width, new_height))
    img = np.array(img)
    return img, png


def simplify_colors(image):
    """
    Simplifies the color representation of an image by converting pixel values.
    All color components are set to either 0 or 255 based on a threshold of 127.

    Args:
    - image (numpy.ndarray): The image array whose colors are to be simplified.

    Returns:
    - numpy.ndarray: The image array with simplified color values.
    """
    image = np.where(image > 127, 255, 0)
    # Flatten the pixels of the image.
    # Each component of each color of each pixel need to be either 0 or 255
    # Values lower than or equal to 127 are converted to 0, higher than 127 to 255
    return image


def compute_search_space_size(file, size):
    """
    Computes the size of the search space for color combinations in both the original and simplified
    images loaded from a file. This helps in understanding the complexity of processing images
    based on the number of possible color combinations.

    Args:
    - file (str): The path to the image file.
    - size (int): The maximum dimension to which the image should be scaled.

    Returns:
    - tuple: A tuple containing the number of possible color combinations in the original image
             and in the simplified image.
    """
    img, png = load_image(file, size)
    simplified = simplify_colors(img)
    num_pixels = img.shape[0] * img.shape[1]
    num_channels = 4 if png else 3
    num_colors_original = 256 ** num_channels  # RGBA vs RGB
    num_colors_simplified = 2 ** num_channels
    combinations_original = num_pixels * num_colors_original
    simple_combinations = num_pixels * num_colors_simplified
    # Compute the size of the search space vs the size of the image
    # comparing the original image stored in file and the simplified one
    # Take into account the number of pixels and the number of colors to compute the total combinations
    # As a suggestion, use a minimum size of 2 and a maximum of 600, step 16
    # Return a tuple containing the number of combinations of the original image and the simplified one
    return combinations_original, simple_combinations


def plot_search_space_size(combinations_png, simple_combinations_png, combinations_jpg, simple_combinations_jpg):
    """
    Plots the comparison of search space sizes for different image types (PNG and JPEG) and formats
    (original and simplified). It includes three plots with varying y-axis limits to provide different
    perspectives on the data.

    Args:
    - combinations_png (list): The number of combinations for original PNG images across different sizes.
    - simple_combinations_png (list): The number of combinations for simplified PNG images across different sizes.
    - combinations_jpg (list): The number of combinations for original JPEG images across different sizes.
    - simple_combinations_jpg (list): The number of combinations for simplified JPEG images across different sizes.
    """
    sizes = np.arange(2, 601, 16)  # Assuming this range is being passed correctly

    # Create a figure with two subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # First plot
    ax1.set_title('Full Range Search Space Size Comparison')
    ax1.set_xlabel('Image size (maximum dimension)')
    ax1.set_ylabel('Number of Combinations')
    ax1.plot(sizes, combinations_png, label='Original PNG', color='blue')
    ax1.plot(sizes, combinations_jpg, label='Original JPEG', color='red')
    ax1.plot(sizes, simple_combinations_png, label='Simplified PNG', color='blue', linestyle = "--")
    ax1.plot(sizes, simple_combinations_jpg, label='Simplified JPEG', color='red', linestyle='--')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Limited Range Search Space Size Comparison')
    ax2.set_xlabel('Image size (maximum dimension)')
    ax2.set_ylabel('Number of Combinations')
    ax2.plot(sizes, combinations_png, label='Original PNG', color='blue')
    ax2.plot(sizes, combinations_jpg, label='Original JPEG', color='red')
    ax2.plot(sizes, simple_combinations_png, label='Simplified PNG', color='blue', linestyle = "--")
    ax2.plot(sizes, simple_combinations_jpg, label='Simplified JPEG', color='red', linestyle='--')
    ax2.set_ylim([0, max(max(combinations_png), max(combinations_jpg)) * 0.05])
    ax2.legend()
    ax2.grid(True)


    # A third plot with very limited y-axis
    ax3.set_title('Very Limited Range Search Space Size Comparison')
    ax3.set_xlabel('Image size (maximum dimension)')
    ax3.set_ylabel('Number of Combinations')
    ax3.plot(sizes, combinations_png, label='Original PNG', color='blue')
    ax3.plot(sizes, combinations_jpg, label='Original JPEG', color="red")
    ax3.plot(sizes, simple_combinations_png, label='Simplified PNG', color='blue', linestyle = "--")
    ax3.plot(sizes, simple_combinations_jpg, label='Simplified JPEG', color='red', linestyle='--')
    ax3.set_ylim([0, max(max(combinations_png), max(combinations_jpg)) * 0.0000001])  # Show the lower 10% of values for detail
    ax3.legend()
    ax3.grid(True)

    plt.show()


def compute_genotypes(image, png):
    """
    Calculates the unique color genotypes in an image and their corresponding probabilities.
    A genotype here refers to a unique color configuration of a pixel, considering all color channels.

    Args:
    - image (numpy.ndarray): The image array from which genotypes are to be computed.
    - png (bool): Indicates if the image uses the PNG format (RGBA) or not (JPEG - RGB).

    Returns:
    - tuple: A tuple containing two elements:
        1. genotypes (numpy.ndarray): An array of unique color genotypes (rows representing unique pixel colors).
        2. probabilities (numpy.ndarray): An array representing the probability of each genotype occurring in the image.
    """
    # First flatten the image array to a 2d array where each row represents a pixel
    if png:
        # RGBA for PNG
        px_array = image.reshape(-1,4)
    else:
        # RGB for JPEG
        px_array = image.reshape(-1,3)
    # Find unique rows and their counts
    genotypes, counts = np.unique(px_array, axis=0, return_counts=True)
    # Calculate total num of pixels
    total_pixels = px_array.shape[0]
    # Calculate probs for each unique color genotype
    probabilities = counts / total_pixels
    return genotypes, probabilities
# Compute the probability of each genotype in the objective image
# In our case, it is the percentage of each color for pixels (remember PNG uses RGBA format, JPEG uses RGB)
# Hint: it is the probability of each color, not each component of each color
# Return the different genotypes and their probabilities


def generate_random_image(image, genotypes, prob):
    """
    Generates a random image based on a predefined set of genotypes and their probabilities.
    This function is typically used to create a random population for genetic algorithms.

    Args:
    - image (numpy.ndarray): The original image array to determine the size of the generated image.
    - genotypes (numpy.ndarray): An array of pixel genotypes from which the new image will be randomly generated.
    - prob (numpy.ndarray): The probabilities associated with each genotype.

    Returns:
    - numpy.ndarray: A randomly generated image of the same size as the original but with pixel values drawn from the genotypes.
    """
    # Size of the image
    image_size = image.shape
    # Flatten image to a list of pixels
    num_pixels = image_size[0] * image_size[1]
    # Randomly select genotypes according to their probs
    indices = np.random.choice(len(genotypes), size=num_pixels, p=prob)
    # Map indices to genotypes to construct image, and reshape the flat array back into original shape.
    random_image_array = genotypes[indices].reshape(image_size[0], image_size[1], -1)
    # Given the objective image, the different genotypes and their probability (prob),
    # Generate a random image with the same size and the same phenotype probability as the objective image
    return random_image_array


def fitness_function(member, goal, png):
    """
    Computes the fitness score of a generated image relative to the goal image by counting the number of differing pixels.

    Args:
    - member (numpy.ndarray): The generated image whose fitness is being evaluated.
    - goal (numpy.ndarray): The target or goal image used as the standard for comparison.
    - png (bool): A flag indicating if the goal image is in PNG format (and thus might have an alpha channel).

    Returns:
    - int: The fitness score, representing the total number of pixel mismatches between the member and the goal images.
    """
    # Compute the fitness between the initial random image and the objective image
    # Just count the different pixels between the two images
    if png and member.shape[-1] == 3:
        member = member[...,:3]
    elif not png and goal.shape[-1] == 4:
        goal = goal[...,:3]
    fitness = np.sum(member != goal)
    return fitness


#-----------------------------------WEEK2----------------------------------------------------------------

def initial_population(goal, n_population, png):
    """
    Generates an initial population of images for genetic algorithms based on the genotypes derived from a goal image.

    Args:
    - goal (numpy.ndarray): The goal or target image from which genotypes are derived.
    - n_population (int): The number of images in the generated population.
    - png (bool): Indicates if the goal image is in PNG format, affecting genotype calculation.

    Returns:
    - list: A list of numpy.ndarray elements, each being a randomly generated image based on the goal's genotypes.
    """
    # Generate the initial population using the objective image (goal)
    # The initial population should be large enough to facilitate the exploration.
    # Set the parameter n_population for the elements in this initial population
    # Hint: use a reasonable value, not too big (long execution), not too small (solution not reachable)
    # The initial population can be created at random or by using a heuristic
    # For session 2, use random generation
    genotypes, probabilities = compute_genotypes(goal, png)
    population = []
    for _ in range(n_population):
        new = generate_random_image(goal, genotypes, probabilities)
        population.append(new)
    return population

def initial_population_improved(goal, n_population, png, n_colors=2):
    """
    Generates an improved initial population of images for genetic algorithms using dominant colors
    found through k-means clustering. This method considers using a set number of dominant colors to
    construct the initial population, potentially improving genetic diversity and convergence speed.

    Args:
    - goal (numpy.ndarray): The target or goal image used to extract dominant colors.
    - n_population (int): The number of images to generate for the initial population.
    - png (bool): Indicates if the image uses the PNG format; affects color handling.
    - n_colors (int): The number of dominant colors to use, determined by k-means clustering.

    Returns:
    - list: A list of numpy.ndarray elements, each a randomly generated image using dominant colors.
    """
    # Find dominant colors using k-means clustering
    dominant_colors, color_weights = find_dominant_colors(goal, n_colors)
    population = []
    for _ in range(n_population):
        # Generate an image by selecting random colors from the dominant set for each pixel
        random_colors_indices = np.random.choice(dominant_colors.shape[0], size=goal.shape[0] * goal.shape[1], p=color_weights)
        random_image_array = dominant_colors[random_colors_indices].reshape(goal.shape[0], goal.shape[1], goal.shape[2])
        population.append(random_image_array)
    return population


def find_dominant_colors(pixels, n_colors):
    """
    Determines the dominant colors in an image using k-means clustering, which groups pixel colors into clusters
    based on their spatial proximity in color space, effectively reducing color diversity to the most representative colors.

    Args:
    - pixels (numpy.ndarray): An array of pixel data from an image.
    - n_colors (int): The number of color clusters to form, representing the main colors of the image.

    Returns:
    - tuple: A tuple containing:
        1. dominant_colors (numpy.ndarray): An array of cluster centers representing dominant colors.
        2. color_weights (numpy.ndarray): The relative frequencies of these colors within the image.
    """
    pixels = pixels.reshape(-1, pixels.shape[-1])
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_, minlength=n_colors)
    color_weights = counts/counts.sum()
    dominant_colors = np.rint(kmeans.cluster_centers_).astype(int)
    return dominant_colors, color_weights


def selection(population, scores, k=10):
    """
    Selects individuals from a population for the next generation using tournament selection,
    which is a method of selecting the best candidate from a randomly chosen subset of the population.

    Args:
    - population (list): A list of individuals from which to select.
    - scores (list): A list of fitness scores corresponding to each individual in the population.
    - k (int): The number of individuals to include in each tournament.

    Returns:
    - list: A list of selected individuals forming the next generation.
    """
    # Initialize the selected population list
    selected_population = []

    # Perform selection process using tournament selection
    for _ in range(len(population)):
        # Select k random indices for the tournament
        tournament_indices = np.random.choice(range(len(population)), size = k, replace = False)
        # Conduct the tournament to find the best individual
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_index = np.argmin(tournament_scores)
        selected_population.append(tournament_individuals[winner_index])

        # Append the best individual from the tournament to the selected population
    return selected_population


def selection_improved(population, scores):
    """
    An improved selection function that ranks individuals based on their fitness scores and
    selects them probabilistically, with a higher chance of selecting fitter individuals.

    Args:
    - population (list): A list of individuals in the population.
    - scores (list): A list of fitness scores for each individual.

    Returns:
    - list: A list of selected individuals based on ranked probabilities.
    """
    ranked_indices = np.argsort(scores)
    ranked_population = [population[i] for i in ranked_indices]
    selection_probabilities = np.linspace(1,0, len(ranked_population))
    selection_probabilities /= selection_probabilities.sum()
    selected_indices = np.random.choice(len(ranked_population), size=len(ranked_population), p = selection_probabilities)
    return [ranked_population[i] for i in selected_indices]


def crossover(p1, p2, r_cross):
    """
    Performs crossover between two parent individuals to produce offspring. The crossover
    is performed at a random point along the rows of the image, mixing the pixel data from
    both parents.

    Args:
    - p1, p2 (numpy.ndarray): Parent images.
    - r_cross (float): The probability that crossover will occur; if not, children are clones of parents.

    Returns:
    - tuple: A tuple containing two numpy.ndarray elements, each an offspring image.
    """
    # From the selected parents p1 and p2, produce the descendants c1 and c2
    # Use r_cross parameter to decide if the crossover is produced or descendants are kept equal to parents
    # Crossover can be in one single middle point, one single random point, multiple random points,
    # multiple equally spaced points, alternating points, etc.
    # Being a 2D matrix, the crossover can be either by column, by row or both
    # For session 2, use one point random crossover
    c1, c2 = np.copy(p1), np.copy(p2)
    if np.random.rand() < r_cross:
        row = np.random.randint(1, p1.shape[0])
        c1[:row,:], c2[:row, :] = p2[:row, :], p1[:row, :]
    return c1, c2

def crossover_improved(p1, p2, r_cross, rate=0.5):
    """
    Performs an improved form of crossover between two parent images, using a mask to decide pixel interchanges.
    This method increases diversity by allowing genes (pixels) from both parents to mix based on a defined rate.

    Args:
    - p1, p2 (numpy.ndarray): The parent images from which offspring are derived.
    - r_cross (float): The probability of crossover occurring.
    - rate (float): The rate at which pixels are swapped between parents; determines the granularity of the crossover.

    Returns:
    - tuple: A tuple containing two offspring images resulting from the crossover operation.
    """
    c1, c2 = np.copy(p1), np.copy(p2)
    if np.random.rand() < r_cross:
        mask = np.random.rand(*p1.shape) < rate
        c1[mask], c2[mask] = p2[mask], p1[mask]
    return c1,c2


def mutation(descendant, r_mut, genotypes, prob):
    """
    Mutates an individual by randomly changing its alleles (pixels) based on a given probability,
    using the provided set of possible genotypes. This adds variability to the population and
    can help escape local minima in optimization problems.

    Args:
    - descendant (numpy.ndarray): The image (individual) to mutate.
    - r_mut (float): The rate at which mutations occur (probability of any pixel mutating).
    - genotypes (numpy.ndarray): The set of possible genotypes (pixel values) that can be chosen during mutation.
    - prob (numpy.ndarray): The probability associated with each genotype being selected during mutation.

    Returns:
    - numpy.ndarray: The mutated image.
    """
    # Mutate a descendant
    # Mutation can be implemented in several ways: bti flip, swap, random, scramble, etc.
    # Mutation should use only the possible genotypes with the given probability prob
    # For session 2, use random mutation of each allele (pixel) with a probability
    mutated_descendant = np.copy(descendant)
    flat_genotypes = genotypes.reshape(-1, genotypes.shape[-1])
    choices = np.arange(len(prob))

    for i in range(mutated_descendant.shape[0]):
        for j in range(mutated_descendant.shape[1]):
            if np.random.rand() < r_mut:
                index = np.random.choice(choices, p=prob)
                mutated_descendant[i,j] = flat_genotypes[index]
            else:
                mutated_descendant[i,j] = descendant[i,j]
    return mutated_descendant


def mutation_improved(descendant, r_mut):
    """
    Applies an improved mutation strategy to an image, where pixel values are determined
    by the mean values of their neighborhoods. This mutation introduces changes that are more
    locally consistent, potentially leading to smoother images.

    Args:
    - descendant (numpy.ndarray): The image to mutate.
    - r_mut (float): The mutation rate, or the probability of any pixel undergoing mutation.

    Returns:
    - numpy.ndarray: The mutated image.
    """
    mutated_descendant = np.copy(descendant)
    rows, cols = descendant.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < r_mut:
                start_row, end_row = max(0,i-1), min(rows,i+2)
                start_col, end_col = max(0,j-1), min(cols,j+2)
                neighborhood = descendant[start_row:end_row, start_col:end_col]
                mean_value = np.mean(neighborhood)
                new_value = 255 if mean_value > 127 else 0
                mutated_descendant[i,j] = new_value
            else:
                mutated_descendant[i,j] = descendant[i,j]
    return mutated_descendant

def replacement(population, descendants, r_replace):
    """
    Replaces the current population with a new set of descendants based on a replacement probability.
    This process determines which individuals carry over to the next generation, ensuring diversity and retention of good traits.

    Args:
    - population (list): The current population of images.
    - descendants (list): The new generation of images to potentially replace the current population.
    - r_replace (float): The probability of replacing any individual in the population with a descendant.

    Returns:
    - list: The new population after replacement decisions.
    """
    # Replacement of the population with the descendants
    # It can be implemented in several ways
    # For session 2, just replace all old population with the new descendants (probability is 100%)
    if r_replace == 1:
        return descendants
    else:
        new_population = []
        for i in range(len(population)):
            if np.random.rand() < r_replace:
                new_population.append(descendants[i])
            else:
                new_population.append(population[i])
    return new_population

def replacement_improved(population, descendants, scores, descendant_scores, retain_top = 0.2):
    """
    Replaces the current population with a mixture of the existing population and the descendants,
    while ensuring that a proportion of the best individuals (elite) are always retained.

    Args:
    - population (list): Current population of individuals.
    - descendants (list): Newly generated descendants that might replace current population members.
    - scores (list): Fitness scores of the current population.
    - descendant_scores (list): Fitness scores of the descendants.
    - retain_top (float): The proportion of the top-performing individuals to retain in the new population.

    Returns:
    - list: The new population after integrating and selecting from both the current population and descendants.
    """
    combined = list(zip(population+descendants, scores+descendant_scores))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    top_elite_cut = int(len(population) * retain_top)
    new_population = [item[0] for item in sorted_combined[:top_elite_cut]]

    while len(new_population) < len(population):
        contenders = random.sample(sorted_combined, 5)
        winner = min(contenders, key = lambda x:x[1])
        new_population.append(winner[0])
    return new_population



def genetic_algorithm(file, n_iter, n_pop, r_cross, r_mut, r_replace, size=64, mutation_switch_threshold=None):
    """
    Runs a genetic algorithm to optimize an image transformation task, aiming to reduce the fitness
    score by evolving an initial population towards a target configuration.

    Args:
    - file (str): Path to the image file to process.
    - n_iter (int): Number of iterations for the genetic algorithm to run.
    - n_pop (int): Size of the population.
    - r_cross (float): Probability of crossover.
    - r_mut (float): Probability of mutation.
    - r_replace (float): Probability of replacing an individual with a descendant.
    - size (int): Dimension size to which the image is scaled.
    - mutation_switch_threshold (int, optional): Iteration threshold after which to switch to an improved mutation function.

    Returns:
    - tuple: A tuple containing the best solution found and its fitness score.
    """
    # Genetic algorithm should:
    # 1. Generate the initial population
    # 2. Start a loop with n_iter iterations
    #    a. Inside the loop, evaluate the fitness of the population and store the best one
    #    b. Make the selection of the parents
    #    c. Crossover each couple of parents to generate new descendants
    #    d. Mutate the descendants to create diversity
    #    e. Replace the old population with the new one (descendants)
    # 3. Return the best solution and the best fitness
    image, png = load_image(file, size)
    population = initial_population_improved(image, n_pop, png)
    goal = simplify_colors(image)
    genotypes, prob = compute_genotypes(goal, png)

    best = None
    best_eval = float("inf")

    for iteration in range(n_iter):
        scores = [fitness_function(individual, goal, png) for individual in population]
        current_best_idx = np.argmin(scores)
        current_best = population[current_best_idx]
        current_best_score = scores[current_best_idx]

        if current_best_score < best_eval:
            best, best_eval = current_best, current_best_score

        if best_eval < 1:
            break

        parents = selection(population, scores)
        descendants = []
        descendants_scores = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                c1, c2 = crossover_improved(parents[i], parents[i+1], r_cross)
                if mutation_switch_threshold and (iteration > mutation_switch_threshold):
                    c1 = mutation_improved(c1, r_mut)
                    c2 = mutation_improved(c2, r_mut)
                else:
                    c1 = mutation(c1, r_mut, genotypes, prob)
                    c2 = mutation(c2, r_mut, genotypes, prob)
                descendants.append(c1)
                descendants.append(c2)
                descendants_scores.append(fitness_function(c1, goal, png))
                descendants_scores.append(fitness_function(c2, goal, png))
        population = replacement(population, descendants, r_replace)

    return best, best_eval




# -------------------------------
file_png = 'logo_gray_scale.png'
file_jpg = 'logo_color.jpeg'

combinations_png = []
simple_png = []
combinations_jpg = []
simple_jpg = []

sizes = np.arange(2,601,16)

for size in sizes:
    combinations_orig_png, combinations_simple_png = compute_search_space_size(file_png, size)
    combinations_orig_jpg, combinations_simple_jpg = compute_search_space_size(file_jpg, size)

    combinations_png.append(combinations_orig_png)
    simple_png.append(combinations_simple_png)
    combinations_jpg.append(combinations_orig_jpg)
    simple_jpg.append(combinations_simple_jpg)

plot_search_space_size(combinations_png, simple_png, combinations_jpg, simple_jpg)

#------------IMPROVEMENT VISUALIZATION FUNCTIONS-----------------
def compare_ga_versions(file, n_iter, n_pop, r_cross, r_mut, r_replace, size=64, mutation_switch_threshold=None):
    # Prepare data structures to hold fitness scores for both versions
    fitness_scores_standard = []
    fitness_scores_improved = []

    # Load image and simplify colors once, as these do not change
    image, png = load_image(file, size)
    goal = simplify_colors(image)
    genotypes, prob = compute_genotypes(goal, png)

    # Initialize populations
    population_standard = initial_population(goal, n_pop, png)
    population_improved = initial_population_improved(goal, n_pop, png, n_colors = 3)  # Assuming 2 dominant colors

    for iteration in range(n_iter):
        # Standard GA iteration
        scores_standard = [fitness_function(ind, goal, png) for ind in population_standard]
        fitness_scores_standard.append(np.mean(scores_standard))
        population_standard = evolve_population(population_standard, scores_standard, genotypes, prob, goal, png, r_cross, r_mut, r_replace)

        # Improved GA iteration
        scores_improved = [fitness_function(ind, goal, png) for ind in population_improved]
        fitness_scores_improved.append(np.mean(scores_improved))
        population_improved = evolve_population_improved(population_improved, scores_improved, genotypes, prob, goal, png, r_cross, r_mut, r_replace, iteration, mutation_switch_threshold)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_iter), fitness_scores_standard, label='Standard GA')
    plt.plot(range(n_iter), fitness_scores_improved, label='Improved GA')
    plt.title('Comparison of Average Fitness Scores per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def evolve_population(population, scores, genotypes, prob, goal, png, r_cross, r_mut, r_replace):
    parents = selection(population, scores)
    descendants = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i+1]
            c1, c2 = crossover(p1, p2, r_cross)
            c1 = mutation(c1, r_mut, genotypes, prob)
            c2 = mutation(c2, r_mut, genotypes, prob)
            descendants.extend([c1, c2])
    return replacement(population, descendants, r_replace)

def evolve_population_improved(population, scores, genotypes, prob, goal, png, r_cross, r_mut, r_replace, iteration, mutation_switch_threshold):
    parents = selection_improved(population, scores)
    descendants = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i+1]
            c1, c2 = crossover_improved(p1, p2, r_cross)
            if mutation_switch_threshold and (iteration > mutation_switch_threshold):
                c1 = mutation_improved(c1, r_mut)
                c2 = mutation_improved(c2, r_mut)
            else:
                c1 = mutation(c1, r_mut, genotypes, prob)
                c2 = mutation(c2, r_mut, genotypes, prob)
            descendants.extend([c1, c2])
    return replacement(population, descendants, r_replace)

compare_ga_versions('logo_gray_scale.png', 700, 300, 0.7, 0.01, 1, size=64, mutation_switch_threshold=600)
compare_ga_versions('logo_color.jpeg', 700, 300, 0.7, 0.01, 1, size=64, mutation_switch_threshold=600)


# For PNG image
# define the total iterations
n_iter = 700
# define the population size
n_pop = 500
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 0.005
# replacement rate
r_replace = 1.0
# perform the genetic algorithm search
mutation_switch_threshold = int(n_iter * 0.80)
best, score = genetic_algorithm(file_png, n_iter, n_pop, r_cross, r_mut, r_replace, size=64, mutation_switch_threshold=mutation_switch_threshold)
plt.imshow(best)
plt.title(f'Best PNG Image - Score: {score}')
plt.show()


# For JPEG image
# define the total iterations
n_iter = 700
# define the population size
n_pop = 500
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 0.0005
# replacement rate
r_replace = 1.0
# perform the genetic algorithm search
mutation_switch_threshold = int(n_iter * 0.80)
best, score = genetic_algorithm(file_jpg, n_iter, n_pop, r_cross, r_mut, r_replace, size=64, mutation_switch_threshold=mutation_switch_threshold)
plt.imshow(best)
plt.title(f'Best JPEG Image - Score: {score}')
plt.show()

