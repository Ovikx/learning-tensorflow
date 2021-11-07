from PIL import Image, ImageOps
import os
import numpy as np

class ImagePreprocessor:
    def __init__(self, pixels=64, normalization=1, training_threshold=1, preprocess_method='square_resize'):
        '''
        Constructs the ImagePreprocessor object

        Args:
            pixels : Integer, number of pixels along each side of the processes image
            normalization : Integer, divisor that squeezes the RGB values
            training_threshold : Float, threshold that determines the proportion of data to be used for training
            preprocess_method : String, "square_resize" squeezes the images into squares and "square_crop" removes excess content to create a square image
        '''
        self.pixels = pixels
        self.normalization = normalization
        self.training_threshold = training_threshold
        self.preprocess_method = preprocess_method
    
    def load_directory_contents(self, path):
        '''
        Returns the images in specified directory

        Args:
            path : String, path where the images are stored
        
        Returns:
            List : PIL image objects
        '''
        return [Image.open(f'{path}/{f}') for f in os.listdir(path)]

    def prepare_image(self, image):
        '''
        Performs the preprocess method on the input image and/or resizes it

        Args:
            image : PIL Image, image to be preprocessed
        
        Returns:
            PIL Image : Preprocessed image
        '''
        if self.preprocess_method == 'square_crop':
            width, height = image.size
            h_cut = v_cut = 0

            if width > height:
                h_cut = (width - height)/2
            elif width < height:
                v_cut = (height - width)/2

            image = ImageOps.crop(image, (h_cut, v_cut, h_cut, v_cut))
        image = image.resize((self.pixels, self.pixels), Image.BILINEAR).convert('RGB')
        return image

    def directory_to_array(self, path):
        '''
        Returns a NumPy array containing image data from specified directory

        Args:
            path : String, path of directory
        
        Returns:
            NumPy array : array representations of images
        '''
        cropped = [self.prepare_image(image) for image in self.load_directory_contents(path)]
        arr = [np.array(image) for image in cropped]
        return np.array(arr)/self.normalization

    def preprocess_dirs(self, paths, labels, partition=False):
        '''
        Returns a dict containing preprocessed images and their respective labels.

        Args:
            paths : List, directories to be preprocessed
            labels : List, labels of each directory
            partition : Boolean, dictates if the images should be partitioned into training and testing data
        
        Returns:
            Dict : Preprocessed and partitioned data
        '''
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
        
        train_features = np.array(train_features)/self.normalization
        train_labels = np.array(train_labels)
        test_features = np.array(test_features)/self.normalization
        test_labels = np.array(test_labels)
        
        package = {
            'TRAIN_IMAGES' : train_features,
            'TRAIN_LABELS' : train_labels,
            'TEST_IMAGES' : test_features,
            'TEST_LABELS' : test_labels
        }

        return package

    def file_to_array(self, path):
        '''
        Returns a NumPy array representation of image specified by path

        Args:
            path : String, path of image to be converted into an array

        Returns:
            NumPy array : array representation of preprocessed input image
        '''
        image = np.array(self.prepare_image(Image.open(path)))
        image = image/self.normalization
        return image