import glob

from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from training.datasets.transform import get_transform


class ImageDataset(Dataset):
    def __init__(self, folder: str, transforms, sample_type: str = "train"):
        self.folder = folder

        self.transforms = transforms

        self.images_A = sorted(
            glob.glob(os.path.join(folder, "%sA" % sample_type) + "/*.*")
        )
        self.images_B = sorted(
            glob.glob(os.path.join(folder, "%sB" % sample_type) + "/*.*")
        )

        self.len_A = len(self.images_A)
        self.len_B = len(self.images_B)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, index):
        path_A = self.images_A[index % self.len_A]
        path_B = self.images_B[index % self.len_B]

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        if self.transforms:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)

        return img_A, img_B


def get_loader(root, sample_type, size, batch_size):

    dataset = ImageDataset(
        folder=root, transforms=get_transform(size), sample_type=sample_type
    )

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
