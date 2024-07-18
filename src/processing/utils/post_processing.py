import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def energie(i, j, homogenous_df, beta):
    """
    Compute the energy of a patch at position i, j
    """
    energies = []
    neighbors = [(i + x, j + y) for x in (-1, 0, 1) for y in (-1, 0, 1) if not (x == 0 and y == 0)]
    # keep rows from homogenous_df with(i,j) from neighbors
    neighbor_patches = homogenous_df[homogenous_df.apply(lambda row: (row['i'], row['j']) in neighbors, axis=1)]
    # get the list of classes from neibors
    neighbor_classes = neighbor_patches['predicted_class'].tolist()
    #select probability_vector from homogenous_df with index i and j
    probability_vector = homogenous_df[(homogenous_df['i'] == i) & (homogenous_df['j'] == j)]['probability_vector'].values[0]
    #for my_class, p in probability_vector: 
    for my_class, p in enumerate(probability_vector): 
        nb_neigboors_diff = 0 
        for neighbor_class in neighbor_classes: 
            if my_class != neighbor_class:
                nb_neigboors_diff += 1
    
        E = -np.log(p) + beta * nb_neigboors_diff
        energies.append(E)
    # return as array
    return np.array(energies)