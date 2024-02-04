import torchvision.transforms as T
import numpy as np


def tensor2image(tensor):
    """
    This is available only when image distribution is among [-1, 1] -> [0, 255]
    The main function of it is transferring the image from tensor to numpy...
    """
    image = (127.5 * (tensor.cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))
    return image1.astype(np.uint8)


def get_transform(size: int = 128):
    transform = [
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    return T.Compose(transform)
