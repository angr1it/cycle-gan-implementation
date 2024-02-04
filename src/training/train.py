import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import Tensor

from training.models.model import CycleGAN
from training.utils.logger import Logger


def train(
    model: CycleGAN,
    loader: DataLoader,
    device,
    save_path: str = "output",
    epoch: int = 0,
    epochs: int = 100,
    logger: Logger = None
):
    try:
        if logger:
            epoch = logger.epoch
            epochs = logger.n_epochs

        if not logger:
            logger = Logger(epochs, len(loader))

        for epoch in range(epoch, epochs):
            for i, batch in enumerate(loader):
                real_A, real_B = batch
                real_A = Variable(real_A.type(Tensor)).cuda(device)
                real_B = Variable(real_B.type(Tensor)).cuda(device)

                model.optimizer_G.zero_grad()

                losses, fake_A, fake_B = model.get_losses_generator(real_A, real_B)
                losses["loss_G"].backward()
                model.optimizer_G.step()

                model.optimizer_D_A.zero_grad()
                loss_D_A = model.get_loss_discriminator("A", real_A, fake_A)
                loss_D_A.backward()
                model.optimizer_D_A.step()

                model.optimizer_D_B.zero_grad()
                loss_D_B = model.get_loss_discriminator("B", real_B, fake_A)
                loss_D_B.backward()
                model.optimizer_D_B.step()

                losses["loss_D"] = loss_D_A + loss_D_B
                logger.log(
                    losses,
                    images={
                        "real_A": real_A,
                        "real_B": real_B,
                        "fake_A": fake_A,
                        "fake_B": fake_B,
                    },
                )

            torch.save(model.generator_A2B.state_dict(), f"{save_path}/generator_A2B.pth")
            torch.save(model.generator_B2A.state_dict(), f"{save_path}/generator_B2A.pth")
            torch.save(
                model.discriminator_A.state_dict(), f"{save_path}/discriminator_A.pth"
            )
            torch.save(
                model.discriminator_B.state_dict(), f"{save_path}/discriminator_B.pth"
            )
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        logger.save_losses()
        logger.viz.save(['main'])

    return logger
