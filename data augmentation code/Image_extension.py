import numpy as np
import random
import PIL
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):

    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=PIL.Image.NEAREST)
        return image, target

class RandomFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class B_S_H(object):
    def __init__(self, BSH_prob):
        self.BSH_prob = BSH_prob

    def __call__(self, image, target):
        if random.random() < self.BSH_prob:
            x = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, x)

            y = random.uniform(0.8, 1.2)
            image = F.adjust_saturation(image, y)

            # z = random.uniform(0.5, 1.5)
            # image = F.adjust_hue(image, z)
        return image, target

class Rotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob

    def __call__(self, image, target):
        if random.random() < self.rotate_prob:
            value = [0, 90, 180, 270]
            r = random.sample(value, 1)
            image = F.rotate(image, angle=r[0])
            target = F.rotate(target, angle=r[0])

        return image, target

