from PIL import Image, ImageOps
import os
import numpy as np

class ImagePreprocessor:
    def __init__(self, pixels=64, normalization=1, training_threshold=1):
        self.pixels = pixels
        self.normalization = normalization
        self.training_threshold = training_threshold
    
    # Returns the images in specified directory
    def load_directory_contents(self, path):
        return [Image.open(f'{path}/{f}') for f in os.listdir(path)]

    # Center crops the image into a square and pixelates it
    def prepare_image(self, image):
        width, height = image.size
        h_cut = v_cut = 0

        if width > height:
            h_cut = (width - height)/2
        elif width < height:
            v_cut = (height - width)/2

        image = ImageOps.crop(image, (h_cut, v_cut, h_cut, v_cut))
        return image.resize((self.pixels, self.pixels), Image.BILINEAR).convert('RGB')

    # Returns a NumPy array containing image data from specified directory
    def directory_to_array(self, path):
        cropped = [self.prepare_image(image) for image in self.load_directory_contents(path)]
        arr = [np.array(image) for image in cropped]
        return np.array(arr)/self.normalization

    # Returns a dict containing features and labels--can be partitioned into training and testing data
    def preprocess_dirs(self, paths, labels, partition=False):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        for label_index, path in enumerate(paths):
            cropped = [self.prepare_image(image) for image in self.load_directory_contents(path)]
            np.random.shuffle(cropped)
            threshold = int(len(cropped)*self.training_threshold)
            dir_images = [np.array(image) for image in cropped]

            for i, image in enumerate(dir_images):
                if i <= threshold or not partition:
                    train_features.append(image)
                    train_labels.append(labels[label_index])
                else:
                    test_features.append(image)
                    test_labels.append(labels[label_index])
        
        train_features = np.array(train_features)/255
        train_labels = np.array(train_labels)
        test_features = np.array(test_features)/255
        test_labels = np.array(test_labels)
        
        package = {
            'TRAIN_IMAGES' : train_features,
            'TRAIN_LABELS' : train_labels,
            'TEST_IMAGES' : test_features,
            'TEST_LABELS' : test_labels
        }

        return package
    
    # Returns a NumPy array representation of image specified by path
    def file_to_array(self, path):
        image = np.array(self.prepare_image(Image.open(path)))
        image = image/self.normalization
        return image