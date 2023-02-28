import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resize(object):
    '''
        Resize the training diagram samples, resize the longest edge as max_size
    '''
    def __init__(self, max_size):
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        if w < h:
            ow = int(w * self.max_size / h)
            oh = self.max_size
        else:
            ow = self.max_size
            oh = int(h * self.max_size / w)
        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

class CenterCrop(object):
    '''
        Crops the given image at the center.
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return F.center_crop(image, self.size)

class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            flip_method = random.choice([0,1,2])
            if flip_method==0:
                image = F.hflip(image)
            elif flip_method==1:
                image = F.vflip(image)
            elif flip_method==2:
                image = F.vflip(F.hflip(image))
        return image

class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)

class Normalize(object):
    def __init__(self, mean=[0.85,0.85,0.85], std=[0.3,0.3,0.3]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image