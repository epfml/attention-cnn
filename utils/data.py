from torch.utils.data import Dataset
import torch
import numpy as np


class MaskedDataset(Dataset):
    """
    Wrap a dataset of images and append a random mask to each sample
    """

    def __init__(self, dataset, mask_size):
        self.dataset = dataset
        self.mask_size = mask_size

    def __getitem__(self, item):
        sample = self.dataset[item]
        image = sample[0]
        _, width, height = image.shape

        batch_mask = torch.ones([width, height], dtype=torch.uint8)
        mask_left = np.random.randint(0, width - self.mask_size)
        mask_top = np.random.randint(0, height - self.mask_size)
        batch_mask[mask_left : mask_left + self.mask_size, mask_top : mask_top + self.mask_size] = 0

        return sample + (batch_mask,)

    def __len__(self):
        return len(self.dataset)

