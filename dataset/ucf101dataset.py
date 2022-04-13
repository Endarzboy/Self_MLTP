
import os
import cv2
import PIL
import glob
import random
import numbers
import argparse
import numpy as np

import skimage.transform

import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore') 

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds

    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return np.asarray(rotated, dtype=np.float32)


class SplitClipChannels(object):
    """Split the RGB channels of each clip randomly
    """

    def __init__(self, channels=None):
        self.channels = channels

    def __call__(self, clip):
        """
        Args:
        clip (PIL.Image or numpy.ndarray): array of frames for channel split
        in format (t, h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Splitted channel (R or G or B) channel
        """
        if self.channels is None:
            self.channels = {7: 'R', 8: 'G', 9: 'B'}

        channel_i = random.choice(list(self.channels))

        frames = []

        r, g, b = None, None, None
        for i in range(clip.shape[0]):
            # b, g, r = cv2.split(clip[i])
            if channel_i == 7:
                _, _, r = cv2.split(clip[i])
                frames.append(r)
            elif channel_i == 8:
                _, g, _ = cv2.split(clip[i])
                frames.append(g)
            else:
                b, _, _ = cv2.split(clip[i])
                frames.append(b) # Frames[0]: (224, 224) -> W, H

        return np.asarray(frames), channel_i, self.channels[channel_i]


class UCF101Dataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, sequence_length, training, rotation_transform=None, crop_transform=None):
        self.training = training
        self.crop_transform = crop_transform
        self.rotation_transform = rotation_transform

        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)

        self.idx2target = {0: 'random', 1: 'center', 2: 'corner', 3: '0', 4: '90', 5: '180', 6: '270', 7: 'R', 8: 'G', 9: 'B'}
        self.label2idx = {v: k for k, v in self.idx2target.items()}

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".avi")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name)]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def _load_rgb_frames(self, image_paths, start_i, sample_interval):
        # Extract frames as numpy array
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                frame = cv2.imread(image_paths[i], 1) # [:, :, [2, 1, 0]]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = (frame/255.)*2 - 1  # Normalize
                image_sequence.append(frame)
        
        return np.asarray(image_sequence, dtype=np.float32)

    def _cropClip(self, frames, crop_type=None):
        
        if crop_type is None:
            crop_type = {0: 'random', 1: 'center', 2: 'corner'}

        crop_i = random.choice(list(crop_type))
        
        cropped_clip = None
        if crop_i == 0 and self.crop_transform is not None:
            cropped_clip = self.crop_transform[crop_i](frames)
        elif crop_i == 1 and self.crop_transform is not None:
            cropped_clip = self.crop_transform[crop_i](frames)
        else:
            cropped_clip = self.crop_transform[crop_i](frames)

        return cropped_clip, crop_i, crop_type[crop_i]

    def _rotateClip(self, frames, rotation_type=None):
        if rotation_type is None:
            rotation_type = {3: '0', 4: '90', 5: '180', 6: '270'}

        degree_i = random.choice(list(rotation_type))

        rotated_clip = None
        if degree_i == 3 and self.rotation_transform is not None:
            rotated_clip = self.rotation_transform[degree_i](frames)
        elif degree_i == 4 and self.rotation_transform is not None:
            rotated_clip = self.rotation_transform[degree_i](frames)
        elif degree_i == 5 and self.rotation_transform is not None:
            rotated_clip = self.rotation_transform[degree_i](frames)
        else:
            rotated_clip = self.rotation_transform[degree_i](frames)

        return rotated_clip, degree_i, rotation_type[degree_i]

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Pad frames sequences shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)
        
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length

        frames = self._load_rgb_frames(image_paths, start_i, sample_interval) 

        #  Step 1: Cropping Transformation  #
        cropped_clip, _, crop_type = self._cropClip(frames)
        
        #  Step 2: Rotation Transformation  #
        rotated_clip, _, rotation_type = self._rotateClip(cropped_clip)
        
        #  Step 3: Channel Split Transforamtion  #
        splited_clip, _, channel_type = SplitClipChannels()(rotated_clip) # (16, 224, 224)
        
        multi_labels = np.zeros((len(list(self.idx2target)),))

        labels = [self.label2idx[crop_type], self.label2idx[rotation_type], self.label2idx[channel_type]]

        
        for i in range(len(labels)):
            if labels[i]==list(self.idx2target)[labels[i]]:
                multi_labels[labels[i]] = 1

        dict_sample = {
            'frames': torch.from_numpy(splited_clip).unsqueeze(0),
            'labels': torch.tensor(multi_labels)
        }

        return dict_sample

    def __len__(self):
        return len(self.sequences)