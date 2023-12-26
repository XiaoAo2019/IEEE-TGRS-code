import PIL
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):

        image = F.resize(image, size=[self.height, self.width])

        target = F.resize(target, size=[self.height, self.width], interpolation=PIL.Image.NEAREST)
        return image, target