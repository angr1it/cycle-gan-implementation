import os
from PIL import Image
import asyncio

import torch
from torch.autograd import Variable
from torch import Tensor
from dotenv import load_dotenv

from training.datasets.transform import get_transform
from training.models.model import CycleGAN
from bot.utils import singleton, tensor2im

load_dotenv()
MODEL_DIR = os.environ.get("MODEL_DIR")


@singleton
class InferenceModel:
    def __init__(self, load_dir: str = MODEL_DIR, n_res_blocks: int = 5) -> None:
        self.device = "cuda"
        self.model = CycleGAN(self.device, load=True, load_dir=load_dir, n_res_blocks=n_res_blocks)

    async def process(self, img_path: str, size: int = 256):
        return await asyncio.create_task(self.__translate(img_path, size))

    async def __translate(self, img_path, size: int = 256):
        transform = get_transform(size)
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            real_A = Variable(transform(img).type(Tensor)).cuda(self.device)
            res = self.model.generator_A2B(real_A)

        return tensor2im(res)
