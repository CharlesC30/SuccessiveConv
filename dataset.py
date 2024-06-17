import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import math
from pathlib import Path
import dataclasses

def extract_patches(image, patch_size, stride):
    patches = []
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            if (i + patch_size <= image.shape[0]) and (j + patch_size <= image.shape[1]):
                patch = image[i: i + patch_size, j: j + patch_size]
                patches.append(patch)
    return patches


def calculate_number_patches(height, width, patch_size, stride):
    n_vertical_patches = math.floor((height - patch_size) / stride) + 1
    n_horizontal_patches = math.floor((width - patch_size) / stride) + 1
    return n_vertical_patches * n_horizontal_patches


def interpolate_sinogram(input_sinogram, n_angles):
    x = np.arange(input_sinogram.shape[1])
    theta = np.arange(0, n_angles, n_angles // input_sinogram.shape[0])
    interp = RegularGridInterpolator((theta, x), input_sinogram, bounds_error=False, fill_value=None)
    
    xg, thetag = np.meshgrid(x, np.arange(n_angles))

    return interp((thetag, xg))


@dataclasses.dataclass
class PatchCollection:
    full_view: list[np.ndarray]
    sparse_view: list[np.ndarray]

class SparseViewSinograms(Dataset):
    def __init__(
            self, 
            input_path: Path,
            filename: str, 
            transform=ToTensor(), 
            downsample_factor=4,
            patch_size=50,
            patch_stride=10,
        ) -> None:
        super().__init__()
        self.input_path = input_path
        self.filename = filename
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # 1000 and 512 are hardcoded from the simulation data I generated
        self.patches_per_sinogram = calculate_number_patches(1000, 512, patch_size, patch_stride)
        
        # these are the patches from the sinogram that is currently loaded and being worked on
        self.working_patches = PatchCollection([], [])  

    def update_patches(self, full_sinogram):
        full_patches = extract_patches(full_sinogram, self.patch_size, self.patch_stride)
        sparse_sinogram = full_sinogram[::self.downsample_factor, :]
        interpolated_sinogram = interpolate_sinogram(sparse_sinogram, full_sinogram.shape[0])
        sparse_patches = extract_patches(interpolated_sinogram, self.patch_size, self.patch_stride)
        self.working_patches.full_view = full_patches
        self.working_patches.sparse_view = sparse_patches

    def __len__(self):
        n_files = len([file for file in self.input_path.iterdir() if file.is_file()])
        return n_files * self.patches_per_sinogram
    
    def __getitem__(self, index):
        # if index == 0:
        #     sinogram_path = self.input_path / Path(f"{self.filename}_{index:03}.npy")
        #     full_sinogram = np.squeeze(np.load(sinogram_path)).astype(np.int32, casting="safe")
        #     self.update_patches(full_sinogram)
        if index % self.patches_per_sinogram == 0:
            i = index // self.patches_per_sinogram
            sinogram_path = self.input_path / Path(f"{self.filename}_{i:03}.npy")
            full_sinogram = np.squeeze(np.load(sinogram_path)).astype(np.float32, casting="safe")
            self.update_patches(full_sinogram)

        patch_index = index % self.patches_per_sinogram

        # TODO: check if `.to(torch.float32)` is unsafe and changes values
        return (self.transform(self.working_patches.sparse_view[patch_index]).to(torch.float32),
                self.transform(self.working_patches.full_view[patch_index]).to(torch.float32))
