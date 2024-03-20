import torch
import numpy as np
import csv
from torch.utils.data import Dataset

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N):
        t = tmin + torch.linspace(0, 1, N) * tmax
        self.vals = torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

class ConditionalSwissroll(Dataset):
    """
    Swissroll dataset with regression targets.
    Data is conditioned on a regression condition, to more or less spread out swissroll.
    """

    def __init__(self, tmin, tmax, multmin, multmax, N):
        t = tmin + torch.linspace(0, 1, N) * tmax
        self.vals = torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T
        self.conds = torch.linspace(multmin, multmax, N)

    def __len__(self):
        return len(self.vals)
    
    def __getitem__(self, i):
        return self.vals[i], self.conds[i]

class StarDataset(Dataset):
    """
    Star-shaped dataset with regression targets as the radius.
    """
    def __init__(self, radius, n_points, n_arms=5, spikeness=0.3, noise_std=0.05, seed=0):
        self.radius = radius
        self.n_points = n_points
        self.n_arms = n_arms
        self.spikeness = spikeness
        self.noise_std = noise_std
        self.seed = seed
        torch.manual_seed(seed)
        self.data, self.radii = self._generate_star(radius, n_points, n_arms, spikeness, noise_std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.radii[idx]

    def _generate_star(self, radius, n_points, n_arms, spikeness, noise_std):
        angles = torch.linspace(0, 2 * np.pi, n_points, dtype=torch.float)
        arm_angles = angles * n_arms

        # Compute the radial distances for the star shape
        r_star = radius * ((1 - spikeness) + spikeness * torch.cos(arm_angles))

        x = r_star * torch.cos(angles)
        y = r_star * torch.sin(angles)
        noise_x = torch.randn_like(x) * noise_std
        noise_y = torch.randn_like(y) * noise_std
        x += noise_x
        y += noise_y
        data = torch.stack([x, y], dim=1)
        radii = radius * torch.ones(n_points, dtype=torch.float)
        # calculate the radii of the actual points
        c = torch.sqrt(x**2 + y**2).unsqueeze(1)

        return data, c

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15, delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]

# Mainly used to discard labels and only output data
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])
