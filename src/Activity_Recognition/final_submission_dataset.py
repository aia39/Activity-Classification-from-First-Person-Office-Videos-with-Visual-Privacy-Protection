import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(self, dataset_path, input_shape, sequence_length):
        self.sequences = self._extract_sequence_paths(dataset_path) # creating a list of directories where the extracted frames are saved
        self.sequence_length = sequence_length # Defining how many frames should be taken per video for training and testing
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ) # This is to transform the datasets to same sizes, it's basically resizing -> converting the image to Tensor image -> then normalizing the image -> composing all the transformation in a single image


    def _extract_sequence_paths(
        self, dataset_path
    ):
        """ Extracts paths to sequences given the specified train / test split """
        lines = sorted(glob.glob(dataset_path + '/*'))
        sequence_paths = []
        for line in lines:
            sequence_paths += [os.path.join(line).replace('\\','/')]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.replace('\\','/').split('/')[-1]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        image_path = image_path.replace('\\','/')
        return int(image_path.split('/')[-1].split('.jpg')[0])

    def _pad_to_length(self, sequence):
        """ Pads the video frames to the required sequence length for small videos"""
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    
    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number 
        image_paths = sorted(glob.glob(sequence_path+'/*.jpg'), key=lambda path: self._frame_number(path))
        # Pad frames of videos shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)
        
        # Start at first frame and sample uniformly over sequence
        start_i = 0
        sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
        flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        label = self._activity_from_path(sequence_path)
        return image_sequence, label

    def __len__(self):
        return len(self.sequences)
