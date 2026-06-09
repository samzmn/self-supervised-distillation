import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Sequence, Callable
import visualize


class DinoDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        return x


class RandomGaussianBlur:
    def __init__(self, 
                 kernel_size: int | Sequence[int] = 3,
                 sigma: int | float | Sequence[float] = (0.1, 2.), 
                 p=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.probability = p

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            return T.GaussianBlur(self.kernel_size, self.sigma)(img)
        return img


class DINOMultiCropDataset(Dataset):
    def __init__(self, dataset: torchvision.datasets.VisionDataset, base_transform, 
                 return_original=False, num_local_crops=6, range_of_scales=0.3):
        self.dataset = dataset
        self.return_original = return_original
        self.num_local_crops = num_local_crops
        self.s = range_of_scales
        self.base_transform = base_transform
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])
        # first global crop
        self.global_transform_1 = T.Compose([
            T.RandomResizedCrop([32], scale=(self.s, 1.0), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(p=1.0),
            base_transform
        ])
        # second global crop
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop([32], scale=(self.s, 1.0), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(p=0.1),
            # T.RandomSolarize(0.2),
            base_transform,
        ])
        # transformation for the local small crops
        self.local_transform = T.Compose([
            T.RandomResizedCrop([32], scale=(0.05, self.s), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(p=0.5),
            base_transform,
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, List[Any]]:
        img = self.dataset[idx]
        assert isinstance(img, Image.Image)

        # 2 global crops
        global_crops = [
            self.global_transform_1(img),
            self.global_transform_2(img)
        ]

        # N local crops
        local_crops = [
            self.local_transform(img)
            for _ in range(self.num_local_crops)
        ]

        if self.return_original:
            out = [self.base_transform(img)]
        else:
            out = []
        out += global_crops + local_crops

        return out


def load_cifar(batch_size = 32, shuffle = True, normal=True, 
               return_original=False, num_local_crops=6, seed = None) -> DataLoader:
    """
    This is a dataset of 60,000 3x32x32 color training images, labeled over 10 categories. See more info at the CIFAR homepage.

    The classes are:
    Label	Description
    0	airplane
    1	automobile
    2	bird
    3	cat
    4	deer
    5	dog
    6	frog
    7	horse
    8	ship
    9	truck
    """
    if seed is not None:
        torch.manual_seed(seed)

    base = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),  # scales to [0, 1]
    ]

    if normal:
        base.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # (x - 0.5) / 0.5 → [-1, 1]

    toTensor = T.Compose(base)

    train_data = torchvision.datasets.CIFAR10(
        root="datasets", train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(
        root="datasets", train=False, download=True)
    
    entire_data = torch.utils.data.ConcatDataset([train_data, test_data])

    data_loader = DataLoader(
        DINOMultiCropDataset(
            DinoDataset(entire_data), base_transform=toTensor, 
            num_local_crops=num_local_crops, return_original=return_original, range_of_scales=0.5), 
        batch_size=batch_size, shuffle=shuffle, 
        num_workers=1, prefetch_factor=1, persistent_workers=True)

    return data_loader

if __name__=="__main__":
    data_loader = load_cifar(batch_size = 16, shuffle = False, normal=False, 
               return_original=True, num_local_crops=6)
    n_cols = 1 + 2 + 6 
    n_rows = 16
    plt.figure(figsize=(n_cols, n_rows))
    for x_batch in data_loader:
        for i in range(16):
            index = i * n_cols
            plt.subplot(n_rows, n_cols, index + 1)
            visualize.plot_image(x_batch[0][i])
            plt.subplot(n_rows, n_cols, index + 2)
            visualize.plot_image(x_batch[1][i])
            plt.subplot(n_rows, n_cols, index + 3)
            visualize.plot_image(x_batch[2][i])
            plt.subplot(n_rows, n_cols, index + 4)
            visualize.plot_image(x_batch[3][i])
            plt.subplot(n_rows, n_cols, index + 5)
            visualize.plot_image(x_batch[4][i])
            plt.subplot(n_rows, n_cols, index + 6)
            visualize.plot_image(x_batch[5][i])
            plt.subplot(n_rows, n_cols, index + 7)
            visualize.plot_image(x_batch[6][i])
            plt.subplot(n_rows, n_cols, index + 8)
            visualize.plot_image(x_batch[7][i])
            plt.subplot(n_rows, n_cols, index + 9)
            visualize.plot_image(x_batch[8][i])
        plt.show()
        break
