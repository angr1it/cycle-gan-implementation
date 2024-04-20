import os
import logging
from PIL import Image
from pathlib import Path


import torch
import numpy as np


logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance

@singleton
class DirsInstance:
    def __init__(self) -> None:
        root_path = Path(".")
        data_path = Path("data")
        results_path = Path("results")
        cache_path = Path("data_cache")

        self.data_dir = root_path / data_path
        self.results_dir = root_path / results_path
        self.cache_dir = root_path / cache_path
        self.cache_real = self.cache_dir / "real"
        self.cache_fake = self.cache_dir / "fake"


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image

        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def create_dirs(client_id):
    dirs_to_check = [
        DirsInstance().data_dir,
        DirsInstance().results_dir,
        DirsInstance().cache_real,
        DirsInstance().cache_fake,
    ]
    DirsInstance().cache_dir.mkdir(exist_ok=True)

    for dir in dirs_to_check:
        dir.mkdir(exist_ok=True)
        if client_id not in os.listdir(dir):
            os.mkdir(dir / client_id)
            logging.info(f"Creating {dir} for {client_id}")
