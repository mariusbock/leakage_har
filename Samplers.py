# ------------------------------------------------------------------------
# Custom Sampler to create unbalanced data.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

"""
Custom Sampler to create unbalanced data 
"""
import torch
import numpy as np

# Custom sampler to choose 50% from a given class a, 25% from a given class b, and the last 25% randomly from all classes
class UnbalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, class_a, class_b):
        self.dataset = dataset
        self.class_a = class_a
        self.class_b = class_b
        self.indices_a = [i for i, (_, label) in enumerate(dataset) if label == class_a]
        self.indices_b = [i for i, (_, label) in enumerate(dataset) if label == class_b]
        self.indices_rest = [i for i, (_, label) in enumerate(dataset)]

    def __iter__(self):
        num_samples = len(self.dataset)
        num_a = num_samples // 2
        num_b = num_samples // 4
        num_rest = num_samples - num_a - num_b

        sampled_indices_a = np.random.choice(self.indices_a, num_a, replace=len(self.indices_a) < num_a)
        sampled_indices_b = np.random.choice(self.indices_b, num_b, replace=len(self.indices_b) < num_b)
        sampled_indices_rest = np.random.choice(self.indices_rest, num_rest, replace=len(self.indices_rest) < num_rest)

        sampled_indices = np.concatenate((sampled_indices_a, sampled_indices_b, sampled_indices_rest))
        np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)

      
# Custom sampler to choose equally from all classes
class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = {}
        for i, (_, label) in enumerate(dataset):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)

    def __iter__(self):
        num_samples = len(self.dataset)
        num_classes = len(self.class_indices)
        num_per_class = num_samples // num_classes

        sampled_indices = []
        for indices in self.class_indices.values():
            sampled_indices.extend(np.random.choice(indices, num_per_class, replace=len(indices) < num_per_class))

        np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)    