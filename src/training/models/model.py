import itertools

import torch
import torch.nn as nn
from torch.optim import Adam

from training.models.generator import Generator
from training.models.discriminator import Discriminator
from training.utils.buffer import ReplayBuffer


class CycleGAN:
    def __init__(
        self,
        device,
        channels: int = 3,
        lr: float = 0.0002,
        load: bool = False,
        load_dir: str = None,
        C1: float = 5.0,
        C2: float = 10.0,
        n_res_blocks: int = 9,
    ) -> None:
        self.C1 = C1
        self.C2 = C2

        self.generator_A2B: nn.Module = Generator(
            features=channels, n_residual_blocks=n_res_blocks
        ).cuda(device)
        self.generator_B2A: nn.Module = Generator(
            features=channels, n_residual_blocks=n_res_blocks
        ).cuda(device)
        self.discriminator_A: nn.Module = Discriminator(channels).cuda(device)
        self.discriminator_B: nn.Module = Discriminator(channels).cuda(device)
        if load:
            if not load_dir:
                raise ValueError

            self.generator_A2B.load_state_dict(
                torch.load(
                    f"{load_dir}/generator_A2B.pth",
                    map_location=torch.device(device),
                )
            )
            self.generator_B2A.load_state_dict(
                torch.load(
                    f"{load_dir}/generator_B2A.pth",
                    map_location=torch.device(device),
                )
            )
            self.discriminator_A.load_state_dict(
                torch.load(
                    f"{load_dir}/discriminator_A.pth",
                    map_location=torch.device(device),
                )
            )
            self.discriminator_B.load_state_dict(
                torch.load(
                    f"{load_dir}/discriminator_B.pth",
                    map_location=torch.device(device),
                )
            )

        self.lr = lr
        self.device = device

        self.criterion_GAN = nn.MSELoss().cuda(device)
        self.criterion_cycle = nn.L1Loss().cuda(device)
        self.criterion_identity = nn.L1Loss().cuda(device)

        self.optimizer_G = Adam(
            itertools.chain(
                self.generator_A2B.parameters(), self.generator_B2A.parameters()
            ),
            lr=self.lr,
            betas=(0.5, 0.999),
        )
        self.optimizer_D_A = Adam(
            self.discriminator_A.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_B = Adam(
            self.discriminator_B.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def get_losses_generator(self, real_A: torch.Tensor, real_B: torch.Tensor):

        same_A = self.generator_B2A(real_A)
        same_B = self.generator_A2B(real_B)

        loss_identity_A = self.criterion_identity(same_A, real_A) * self.C1
        loss_identity_B = self.criterion_identity(same_B, real_B) * self.C1

        fake_B = self.generator_A2B(real_A)
        pred_fake = self.discriminator_B(fake_B)
        loss_A2B = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        fake_A = self.generator_A2B(real_B)
        pred_fake = self.discriminator_A(fake_A)
        loss_B2A = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        recovered_A = self.generator_B2A(fake_B)
        loss_A2B2A = self.criterion_cycle(recovered_A, real_A) * self.C2

        recovered_B = self.generator_A2B(fake_A)
        loss_B2A2B = self.criterion_cycle(recovered_B, real_B) * self.C2

        return (
            {
                "loss_G": loss_identity_A
                + loss_identity_B
                + loss_A2B
                + loss_B2A
                + loss_A2B2A
                + loss_B2A2B,
                "loss_G_identity": (loss_identity_A + loss_identity_B),
                "loss_G_GAN": (loss_A2B + loss_B2A),
                "loss_G_cycle": (loss_A2B2A + loss_B2A2B),
            },
            fake_A,
            fake_B,
        )

    def get_loss_discriminator(
        self, discriminator_type: str, real: torch.Tensor, fake: torch.Tensor
    ):
        if discriminator_type not in ["A", "B"]:
            raise ValueError

        discriminator = (
            self.discriminator_A if discriminator_type == "A" else self.discriminator_B
        )
        buffer = self.fake_A_buffer if discriminator_type == "A" else self.fake_B_buffer

        pred_real = discriminator(real)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        fake = buffer.process(fake)
        pred_fake = discriminator(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        return (loss_D_real + loss_D_fake) * 0.5
