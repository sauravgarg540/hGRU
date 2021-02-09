import torch
import torchvision
import numpy as np
from PIL import Image

class Resize(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' as the paper suggest 
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size = 75, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return torchvision.transforms.Resize(self.size, self.interpolation)(image)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) to numpy.ndarray (H x W) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, to_numpy = True):
        self.to_numpy = to_numpy

    def __call__(self, image):
        if self.to_numpy:
            # handle numpy array
            image = np.array(image)
            image = np.expand_dims(image, 0)
            # print(image.shape)
            image = torch.from_numpy(image).contiguous()

        return image.float().div(255.0)
