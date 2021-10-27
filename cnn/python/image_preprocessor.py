from PIL import Image, ImageOps
import os
import numpy as np

class ImagePreprocessor:
    def __init__(self, path=None, pixels=64, label=None, normalize=False):
        self.path = path
        self.images = [Image.open(f'{self.path}/{f}') for f in os.listdir(path)]
        self.pixels = pixels
        self.label = label
        self.size = len(self.images)
        self.normalize = normalize
    
    # Sets the pointer to a new path (useful for loading different datasets)
    def open(self, path=None, label=None):
        self.path = path
        self.images = [Image.open(f'{self.path}/{f}') for f in os.listdir(path)]
        self.label = label

    # Crops the image into a square (centered) and pixelates it
    def prep_image(self, image, show=False):
        width, height = image.size
        h_cut = v_cut = 0

        if width > height:
            h_cut = (width - height)/2
        elif width < height:
            v_cut = (height - width)/2

        image = ImageOps.crop(image, (h_cut, v_cut, h_cut, v_cut))
        image = image.resize((self.pixels, self.pixels), Image.BILINEAR).convert('RGB')
        if show:
            image.show()
        return image

    # Returns a NumPy array with processed images in self.path represented as NumPy arrays
    def dir_to_array(self):
        cropped = [self.prep_image(image) for image in self.images]
        arr = np.array([np.array(image) for image in cropped])
        if self.normalize:
            arr = arr/255
        return arr

    # Returns a zip with NumPy array representations of images in self.path and their respective labels
    def dir_to_zip(self):
        features = self.dir_to_array()
        return zip(features, [self.label for i in range(len(features))])
    
    # Returns a NumPy array representation of path-specified image
    def file_to_array(self, path):
        image = np.array(self.prep_image(Image.open(path)))
        if self.normalize:
            image = image/255
        return image