# ------------------------------------------------------------------------
# Custom Sampler to create unbalanced data.
# ------------------------------------------------------------------------
# Authored by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import torch
import numpy as np

# Custom sampler to choose 90% from a given class and the last 10% randomly from all classes
class DefenseSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, main_class):
        self.dataset = dataset
        self.main_class = main_class
        self.indices_main = [i for i, (_, label) in enumerate(dataset) if label == main_class]
        self.indices_rest = [i for i, (_, label) in enumerate(dataset) if label != main_class]

    def __iter__(self):
        num_samples = len(self.dataset)
        num_main = int(num_samples * 0.6)
        num_rest = num_samples - num_main

        sampled_indices_main = np.random.choice(self.indices_main, num_main, replace=len(self.indices_main) < num_main)
        sampled_indices_rest = np.random.choice(self.indices_rest, num_rest, replace=len(self.indices_rest) < num_rest)

        sampled_indices = np.concatenate((sampled_indices_main, sampled_indices_rest))
        np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)

          