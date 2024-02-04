import time
import datetime
import sys
import json

from visdom import Visdom
import numpy as np


from training.datasets.transform import tensor2image

# As a basis:
# https://github.com/Tloops/DLToolkit/blob/ac5234d3aaf2f76167151f9bbadcd5c4e5dcc5fa/code/vizlogger.py#L84


class Logger:
    """
    a wrapper class for visdom providing basic display functions for training
    """

    def __init__(self, n_epochs: int, batches_epoch: int, epoch: int = 1):
        self.viz = Visdom(server="http://visdom", port=8097)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = epoch
        self.start_epoch = epoch
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

        self.dump_losses = []

    def get_losses(self):
        return self.losses, self.loss_windows

    def save_losses(self, path: str = "./output/losses"):
        with open(f"{path}/losses.json", "w+") as f:
            f.write(json.dumps(self.dump_losses))
        # with open(f"{path}/loss_windows.json", "w+") as f:
        #     f.write(json.dumps(self.loss_windows))

    def load_losses(self, path: str = "./output/losses"):
        try:
            with open(f"{path}/losses.json", "r") as f:
                self.dump_losses = json.loads(f.read())
            # with open(f"{path}/loss_windows.json", "r") as f:
            #     self.loss_windows = json.loads(f.read())
        except Exception:
            print(f"Couldn't find any losses in {path}")
            self.losses, self.loss_windows = {}, {}

        if len(self.dump_losses) > 0:
            for batch, epoch, losses in self.dump_losses:
                self.plot_losses(losses, batch, epoch)

    def plot_losses(self, losses, batch, epoch):
        for loss_name, loss in losses.items():
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(
                    X=np.array([epoch]),
                    Y=np.array([loss / batch]),
                    opts={
                        "xlabel": "epochs",
                        "ylabel": loss_name,
                        "title": loss_name,
                    },
                )
            else:
                self.viz.line(
                    X=np.array([epoch]),
                    Y=np.array([loss / batch]),
                    win=self.loss_windows[loss_name],
                    update="append",
                )

    def log(self, losses=None, images=None):
        """
        :param losses: dictionary for name and value of each loss
        :param images: dictionary for image to be visualized
        """
        self.mean_period += time.time() - self.prev_time
        self.prev_time = time.time()

        sys.stdout.write(
            "\rEpoch %03d/%03d [%04d/%04d] -- "
            % (self.epoch, self.n_epochs, self.batch, self.batches_epoch)
        )

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write(
                    "%s: %.4f -- " % (loss_name, self.losses[loss_name] / self.batch)
                )
            else:
                sys.stdout.write(
                    "%s: %.4f | " % (loss_name, self.losses[loss_name] / self.batch)
                )
        batches_done = self.batches_epoch * (self.epoch - self.start_epoch) + self.batch
        batches_left = (
            self.batches_epoch * (self.n_epochs - self.epoch)
            + self.batches_epoch
            - self.batch
        )
        sys.stdout.write(
            "ETA: %s"
            % (
                datetime.timedelta(
                    seconds=batches_left * self.mean_period / batches_done
                )
            )
        )

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(
                    tensor2image(tensor.data), opts={"title": image_name}
                )
            else:
                self.viz.image(
                    tensor2image(tensor.data),
                    win=self.image_windows[image_name],
                    opts={"title": image_name},
                )

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            self.plot_losses(self.losses, self.batch, self.epoch)
            self.dump_losses.append((self.batch, self.epoch, dict(self.losses)))
            # Reset losses for next epoch
            for loss_name in losses.keys():
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write("\n")
        else:
            self.batch += 1
