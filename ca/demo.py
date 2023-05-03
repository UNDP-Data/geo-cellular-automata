import numpy as np
import matplotlib.pyplot as plt

# Set the size of the forest
n = 50
m = 50

# Initialize the forest with trees and empty cells
elaccess = np.zeros((n, m))
p_tree = 0.6  # probability of a cell being a tree
elaccess[np.random.random((n, m)) > p_tree] = 1

# Set the probability of a tree catching fire
p_fire = 0.01

# Set the probability of a tree regrowing after being burned
p_regrow = 0.01

# Define the update function for each time step
def update(forest):
    # Copy the forest to avoid updating cells multiple times
    new_elaccess = np.copy(elaccess)
    for i in range(n):
        for j in range(m):
            # If the cell is on fire, it becomes empty
            if elaccess[i, j] == 2:
                new_elaccess[i, j] = 0
            # If the cell is a tree and has a burning neighbor, it catches fire
            elif elaccess[i, j] == 1 and (i > 0 and forest[i-1, j] == 2 or
                                        i < n-1 and forest[i+1, j] == 2 or
                                        j > 0 and forest[i, j-1] == 2 or
                                        j < m-1 and forest[i, j+1] == 2):
                if np.random.random() > p_fire:
                    new_elaccess[i, j] = 2
            # If the cell is empty, it may regrow a tree
            elif elaccess[i, j] == 0:
                if np.random.random() < p_regrow:
                    new_elaccess[i, j] = 1
    return new_elaccess

# Define the main simulation loop
def simulate(forest, steps):
    fig, ax = plt.subplots()
    for i in range(steps):
        forest = update(forest)
        ax.imshow(forest, cmap='viridis')
        ax.set_title(f'Time step {i}')
        plt.pause(.8)
    plt.show()

# Run the simulation for 100 time steps
simulate(elaccess, 10)
