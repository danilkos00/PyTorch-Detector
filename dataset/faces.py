import kagglehub
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_convert
import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data


def _load_images():
    path = kagglehub.dataset_download("fareselmenshawii/face-detection-dataset")
    return path

def _get_transforms(train_or_val):
    if train_or_val == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=300),
            A.PadIfNeeded(min_height=300, min_width=300, border_mode=0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.7, 1.3), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.GridDistortion(distort_limit=0.1, p=0.5),
            A.CLAHE(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.MotionBlur(p=0.5),
            A.ISONoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    return A.Compose([
        A.LongestMaxSize(max_size=300),
        A.PadIfNeeded(min_height=300, min_width=300, border_mode=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


class FacesDataset(data.Dataset):
    def __init__(self, train=True, return_original_img=False):
        train_or_val = 'train' if train else 'val'
        data_path = _load_images()
        self.return_original_img = return_original_img
        self.images = glob(os.path.join(data_path, 'images', train_or_val, '*.jpg'))
        self.labels = []

        for i in self.images:
            i = i.split('/')[-1][:-4]
            self.labels.append(os.path.join(data_path, 'labels2', i + '.txt'))

        self.transforms = _get_transforms(train_or_val)

        if return_original_img:
            self.original_transforms = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        bboxes = []
        with Image.open(self.images[idx]) as img:
            original_width = img.size[0]
            original_height = img.size[1]
            image = np.array(img.convert('RGB'))

        with open(self.labels[idx], 'r') as f:
            for line in f.readlines():
                x1, y1, x2, y2 = map(float, line.split()[2:])
                x1 /= original_width
                x2 /= original_width
                y1 /= original_height
                y2 /= original_height

                bboxes.append([x1, y1, x2, y2])

        bboxes = box_convert(torch.tensor(bboxes, dtype=torch.float32), 'xyxy', 'cxcywh').numpy()
        classes = np.ones(bboxes.shape[0])

        data = self.transforms(image=image, bboxes=bboxes, labels=classes)
        while len(data['bboxes']) < 1:
            data = self.transforms(image=image, bboxes=bboxes, labels=classes)   

        if self.return_original_img:
            original_image = self.original_transforms(image=image)['image']

            return data['image'], torch.tensor(data['bboxes'], dtype=torch.float32), torch.tensor(classes, dtype=torch.long), original_image
        
        return data['image'], torch.tensor(data['bboxes'], dtype=torch.float32), torch.tensor(classes, dtype=torch.long)