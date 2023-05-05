import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
import random
import albumentations as A
import glob
from omegaconf import OmegaConf

config = OmegaConf.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'config/test_ours.yaml'))


class P3M10KBlurred(data.Dataset):
    def __init__(self, config=config['datasets']):
        super(P3M10KBlurred, self).__init__()
        dataset_path = config['p3m10k']['path']

        self.bg_files = sorted(glob.glob(config['bg']+'/*.jpg'))
        self.original_files = sorted(
            glob.glob(dataset_path+'/train/blurred_image/*.jpg'))
        self.mask_path = sorted(glob.glob(dataset_path+'/train/mask/*.png'))
        if config['p3m10k']['transform'] == 'hard':
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=1080, interpolation=1,
                                 always_apply=True, p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    rotate_limit=5, scale_limit=0.05, shift_limit=0.02, p=.5),
                A.Blur(blur_limit=3),
                A.PadIfNeeded(config.image_crop, config.image_crop),
                A.RandomCrop(width=config.image_crop,
                             height=config.image_crop),
                A.HueSaturationValue(),
                A.Normalize(),
                ToTensorV2()])
        else:
            self.transform = A.Compose([A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=8, pad_width_divisor=8),
                                        A.Normalize(),
                                        ToTensorV2()
                                        ])

    def __getitem__(self, idx):
        image_file = self.original_files[idx]

        image = np.array(Image.open(image_file).convert('RGB'))

        mask_file = image_file.replace('foregrounds', 'mask')[:-3]+'png'
        mask = np.array(Image.open(mask_file).convert('L'))
        transformed = self.transform(image=image, mask=mask)

        return transformed['image'], transformed['mask'] / 255.

    def __len__(self):
        return len(self.original_files)


class AM2K(data.Dataset):
    def __init__(self, config=config, is_train=True, fake_bg=True):
        super(AM2K, self).__init__()

        if is_train:
            self.original_files = sorted(
                glob.glob(config['am2k']['train_original']+'/*.jpg'))
            self.mask_path = config['am2k']['train_mask']
            masks_path = glob.glob(config['am2k']['train_mask']+'/*.png')
            masks_names = set(
                [path.split('/')[-1][:-3]+'jpg' for path in masks_path])

            # there is semgentations in fg folder for them in original dataset
            self.original_files_no_masks = []
            for files in self.original_files:
                if files.split('/')[-1] not in masks_names:
                    self.original_files_no_masks.append(files)
            self.original_files = sorted(
                list(set(self.original_files)-set(self.original_files_no_masks)))

        else:
            self.original_files = sorted(
                glob.glob(config['am2k']['validation_original']+'/*.jpg'))
            self.mask_path = config['am2k']['validation_mask']

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=1080, interpolation=1,
                             always_apply=True, p=1),
            A.CLAHE(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                rotate_limit=5, scale_limit=0.05, shift_limit=0.02, p=.5),
            A.Blur(blur_limit=3),
            A.PadIfNeeded(config.image_crop, config.image_crop),
            A.RandomCrop(width=config.image_crop, height=config.image_crop),

        ])

        if fake_bg:
            self.bg_list = sorted(
                glob.glob(config['am2k']['background']+'/*.jpg'))
            self.bg_transform = A.Compose([
                A.LongestMaxSize(max_size=1080, interpolation=1,
                                 always_apply=True, p=1),
                A.CLAHE(),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    rotate_limit=5, scale_limit=0.05, shift_limit=0.02, p=.5),
                A.Blur(blur_limit=5),
                A.PadIfNeeded(config.image_crop, config.image_crop),
                A.RandomCrop(width=config.image_crop,
                             height=config.image_crop),
            ])
        else:
            self.bg_list = None

        self.final_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        image_file = self.original_files[idx]
        file_name = image_file.split('/')[-1]

        image = np.array(Image.open(image_file).convert('RGB'))
        mask_file = self.mask_path + '/' + file_name.replace('jpg', 'png')
        mask = np.array(Image.open(mask_file).convert('L'))
        transformed = self.transform(image=image, mask=mask)
        if self.bg_list is not None:
            background_img = np.array(Image.open(
                random.choice(self.bg_list)).convert('RGB'))
            background = self.bg_transform(image=background_img)['image']
            mask = (transformed['mask'] / 255.)[:, :, None]
            composite = np.uint8(
                (transformed['image']*mask+background*(1-mask)))
            final_transformed = self.final_transform(
                image=composite, mask=transformed['mask'])

            return final_transformed['image'], final_transformed['mask'] / 255.
        else:
            final_transformed = self.final_transform(
                image=transformed['image'], mask=transformed['mask'])

            return final_transformed['image'], final_transformed['mask'] / 255.

    def __len__(self):
        return len(self.original_files)


if __name__ == '__main__':
    ds = AM2K(config=config['datasets'])
    print(f'Dataset length: {len(ds)}')
