from os.path import splitext
from os import listdir
from os import path as ospath
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def _get_image_size(cls, dataset):
        """
        Returns a size for images of the given dataset in form of (height, width) tuple.

        :param dataset: Currently supported DRIVE, CHASE, STARE, HRF or DROPS.
        """
        if dataset.lower() == 'drive':
            return 584, 565
        if dataset.lower() == 'chase':
            return 960, 999
        if dataset.lower() == 'stare':
            return 605, 700
        if dataset.lower() == 'drops':
            return 480, 640
        if dataset.lower() == 'hrf':
            return 2336, 3504

    @classmethod
    def _crop_to_shape(cls, data, new_shape):
        if type(data) == np.array:
            return cls._numpy_crop_to_shape(data, new_shape)
        else:
            return cls._pillow_crop_to_shape(data, new_shape)

    @classmethod
    def _pillow_crop_to_shape(cls, data, new_shape):
        """
        Crops a pillow Image object relatively to bottom right corner of a cropping rectangle.

        :param data: The pillow Image to crop.
        :param new_shape: A (height, width) tuple representing the target's shape. Must be smaller of equal to the data
                          shape.

        :returns: Resized pillow Image object.
        """
        # assert type(data) == Image, f"Expects a pillow Image object, but the data is of type {type(data)}."

        w, h = data.size
        new_w, new_h = new_shape[1], new_shape[0]

        if w == new_w and h == new_h:
            return data

        assert new_w <= w and new_h <= h
        left, upper = new_w - w, new_h - h
        right, bottom = left + new_w, upper + new_h

        return data.crop((left, upper, right, bottom))

    @classmethod
    def _numpy_crop_to_shape(cls, data, new_shape):
        """
        Crops the array to the given image shape. The resulting image's will original data in range (0, new_height) to
        (0, new_width). The function expects a numpy array of size [1, h, w, c], [h, w, c] or [h, w]. A shape to resize to
        must be of same dimensionality as the input data array. If data and target shapes are identical, input data is not
        modified.

        :param data: The array to crop.
        :param shape: The target shape. Must be smaller or equal to the source shape.

        :returns: 4D, 3D or 2D cropped numpy array of target shape. If target shape and data shape are identical, unmodified
                  data array is returned.
        """

        if len(data.shape) == 4:
            assert len(new_shape) == 4, f'new_shape is {new_shape}, but should be {data.shape}.'
            if data.shape[1:3] == new_shape[1:3]:  # no need to resize, width and hight are already same
                return data
            assert new_shape[1] <= data.shape[1] and new_shape[2] <= data.shape[2]  # resulting shape must be smaller
            print(f'> [crop] Crop from shape {data.shape} to {new_shape}.')
            return data[:, (data.shape[0] - new_shape[0]):, (data.shape[1] - new_shape[1]):, :]
        elif len(data.shape) == 3:
            assert len(new_shape) == 3, f'new_shape is {new_shape}, but should be {data.shape}.'
            if data.shape[:2] == new_shape[:2]:
                return data
            assert new_shape[0] <= data.shape[0] and new_shape[1] <= data.shape[1]
            print(f'> [crop] Crop from shape {data.shape} to {new_shape}.')
            return data[(data.shape[0] - new_shape[0]):, (data.shape[1] - new_shape[1]):, :]
        elif len(data.shape) == 2:
            assert len(new_shape) == 2, f'new_shape is {new_shape}, but should be {data.shape}.'
            if data.shape == new_shape:
                return data
            assert new_shape[0] <= data.shape[0] and new_shape[1] <= data.shape[1]
            print(f'> [crop] Crop from shape {data.shape} to {new_shape}.')
            return data[(data.shape[0] - new_shape[0]):, (data.shape[1] - new_shape[1]):]

    @classmethod
    def preprocess(cls, pil_img, scale, mode='train', dataset=None):
        w, h = pil_img.size

        assert dataset is not None, 'Dataset must be specified!'
        expected_h, expected_w = cls._get_image_size(dataset)
        if w != expected_w or h != expected_h:
            pil_img = cls._crop_to_shape(pil_img, (expected_h, expected_w))

        scaled_w, scaled_h = int(scale * w), int(scale * h)
        assert scaled_w > 0 and scaled_h > 0, 'Scale is too small'
        pil_img = pil_img.resize((scaled_w, scaled_h))

        img_nd = np.array(pil_img, dtype=np.uint8)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(ospath.join(self.masks_dir, idx + self.mask_suffix + '.*'))
        img_file = glob(ospath.join(self.imgs_dir, idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
