"""
Dataset classes for ILS-HED.
Contains loaders for BSDS500, DRIVE, STARE, DeepCrack, Stone331, SDNet2018, etc.
"""

import os
import numpy as np
import cv2
from scipy.io import loadmat
from skimage import io
from glob import glob
from PIL import Image


class BSDS500Dataset:
    def __init__(self, config, split: str = 'train'):
        self.config = config
        self.split = split
        self.samples = []

        img_dir = os.path.join(config.bsds500_root, 'images', split)
        gt_dir = os.path.join(config.bsds500_root, 'ground_truth', split)

        if not os.path.exists(img_dir):
            print(f"BSDS500 {split} not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, base + '.mat')
            self.samples.append({'image': img_path, 'gt': gt_path if os.path.exists(gt_path) else None, 'name': base})

        print(f"BSDS500 {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            try:
                mat = loadmat(sample['gt'])
                if 'groundTruth' in mat:
                    boundaries = []
                    for i in range(mat['groundTruth'].shape[1]):
                        item = mat['groundTruth'][0, i]
                        if 'Boundaries' in item.dtype.names:
                            boundaries.append(item['Boundaries'][0, 0].astype(np.float32))
                    if boundaries:
                        gt = np.maximum.reduce(boundaries)
                        gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass

        return img, gt, sample['name'], orig


class DeepCrackDataset:
    def __init__(self, config):
        self.config = config
        self.samples = []

        img_dir = os.path.join(config.deepcrack_root, 'image')
        gt_dir = os.path.join(config.deepcrack_root, 'ground_truth')

        if not os.path.exists(img_dir):
            print("DeepCrack not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.JPG'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, base + '.bmp')
            self.samples.append({'image': img_path, 'gt': gt_path if os.path.exists(gt_path) else None, 'name': base})

        print(f"DeepCrack: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            gt_img = io.imread(sample['gt'], as_gray=True)
            gt = (gt_img > 0).astype(np.float32)
            gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)

        return img, gt, sample['name'], orig


class Stone331Dataset:
    def __init__(self, config):
        self.config = config
        self.samples = []

        masks = {}
        if os.path.exists(config.stone331_mask_root):
            for mask_file in glob(os.path.join(config.stone331_mask_root, '*.bmp')):
                base = os.path.splitext(os.path.basename(mask_file))[0]
                masks[base] = mask_file

        for img_path in sorted(glob(os.path.join(config.stone331_root, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'gt': masks.get(base), 'name': base})

        print(f"Stone331: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            gt_img = io.imread(sample['gt'], as_gray=True)
            gt = (gt_img > 0).astype(np.float32)
            gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)

        return img, gt, sample['name'], orig


class CrackLS315Dataset:
    def __init__(self, config):
        self.config = config
        self.samples = []

        for img_path in sorted(glob(os.path.join(config.crackls315_root, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"CrackLS315: {len(self.samples)} images (no GT)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, None, sample['name'], orig


class CRKWH100Dataset:
    def __init__(self, config):
        self.config = config
        self.samples = []

        for img_path in sorted(glob(os.path.join(config.crkwh100_root, '*.png'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"CRKWH100: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, None, sample['name'], orig


class SDNETDataset:
    def __init__(self, config, category: str = 'Decks'):
        self.config = config
        self.samples = []

        cracked_dir = os.path.join(config.sdnet_root, category, 'Cracked')

        if os.path.exists(cracked_dir):
            for img_file in os.listdir(cracked_dir):
                if img_file.lower().endswith(('.jpg', '.png')):
                    self.samples.append({
                        'path': os.path.join(cracked_dir, img_file),
                        'name': img_file,
                        'category': category
                    })

        print(f"SDNET {category}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['path'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, sample['name'], orig, sample['category']


class DRIVEDataset:
    def __init__(self, config, split: str = 'test'):
        self.config = config
        self.split = split
        self.samples = []

        if split == 'test':
            img_dir = os.path.join(config.drive_root, 'test', 'images')
            mask_dir = os.path.join(config.drive_root, 'test', 'mask')
            gt_dir = None
        else:
            img_dir = os.path.join(config.drive_root, 'training', 'images')
            mask_dir = os.path.join(config.drive_root, 'training', 'mask')
            gt_dir = os.path.join(config.drive_root, 'training', '1st_manual')

        if not os.path.exists(img_dir):
            print(f"DRIVE {split} not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.tif')) + glob(os.path.join(img_dir, '*.tiff'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            base_clean = base.replace('_training', '').replace('_test', '')

            mask_path = os.path.join(mask_dir, f"{base_clean}_mask.gif") if mask_dir else None
            if not os.path.exists(mask_path):
                mask_path = None

            gt_path = os.path.join(gt_dir, f"{base_clean}_manual1.gif") if gt_dir else None
            if gt_path and not os.path.exists(gt_path):
                gt_path = None

            self.samples.append({'image': img_path, 'mask': mask_path, 'gt': gt_path, 'name': base_clean})

        print(f"DRIVE {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image with PIL for LZW compression
        img_pil = Image.open(sample['image'])
        img = np.array(img_pil)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        if img.max() > 1:
            img = img / 255.0

        # Load FOV mask
        fov_mask = np.ones(img.shape[:2], dtype=bool)
        if sample['mask']:
            mask = io.imread(sample['mask'])
            fov_mask = (mask > 0)

        # Load GT
        gt = np.zeros(img.shape[:2], dtype=np.float32)
        if sample['gt']:
            gt_img = io.imread(sample['gt'])
            gt = (gt_img > 0).astype(np.float32)

        # Resize
        img = cv2.resize(img, self.config.drive_size, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, self.config.drive_size, interpolation=cv2.INTER_NEAREST)
        fov_mask = cv2.resize(fov_mask.astype(np.uint8), self.config.drive_size,
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        return img, gt, fov_mask, sample['name']


class STAREDataset:
    def __init__(self, config):
        self.config = config
        self.samples = []

        # STARE has images in root directory
        for img_path in sorted(glob(os.path.join(config.stare_root, '*.ppm'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"STARE: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.stare_size, interpolation=cv2.INTER_LINEAR)
        return img, sample['name'], orig
