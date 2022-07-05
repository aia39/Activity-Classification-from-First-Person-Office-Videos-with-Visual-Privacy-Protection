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
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(split_path) #creating a dictionary that has action name as the key and action number as value
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training) # creating a list of directories where the extracted frames are saved
        self.sequence_length = sequence_length # Defining how many frames should be taken per video for training and testing
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences]))) #Getting the label names or name of the class
        self.num_classes = len(self.label_names) # Getting the number of class
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ) # This is to transform the datasets to same sizes, it's basically resizing -> converting the image to Tensor image -> then normalizing the image -> composing all the transformation in a single image

    def _extract_label_mapping(self, split_path="data/viptrainlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
       # for li in range(30):
            #line = lines[li]
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/viptrainlist", split_number=1, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = "trainlist001.txt" if training else "testlist001.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        #for li in range(30):
            #line = lines[li]
        for line in lines:
            seq_name = line.split(".MP4")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name).replace('\\','/')]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.replace('\\','/').split('/')[-2]

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
        if self.training:
            if len(image_paths) == 40:
                # Fixing the start and interval for videos containing only forty frames or as the sequence length
                start_i = 0
                sample_interval = 1
                flip = np.random.random() < 0.5
            else:
                # Randomly choose sample interval and start frame
                sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
                start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
                flip = np.random.random() < 0.5
        else:
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
        target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
