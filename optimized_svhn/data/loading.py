import torch
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.transforms.v2 import Resize, Compose, Normalize, ToDtype, RandomAffine, ColorJitter, RandomApply, GaussianBlur, RandomPerspective
import torchvision.transforms.functional as F
from torchvision.ops import box_convert


class SVHNDataset(Dataset):

    def __init__(
            self,
            folder_path: Path,
            img_size: Tuple[int, int],
            cj_brightness: float,
            cj_contrast: float,
            cj_saturation: float,
            aff_deg: Tuple[int, int],
            aff_trans: Tuple[float, float],
            aff_scale: Tuple[float, float],
            aff_shear: float,
            blur_kernel: int,
            blur_sigma: Tuple[float, float],
            perspective_dist: float,
            augment_prob: float,
    ):
        super().__init__()
        self._bbox_pad_value = 0.0
        self._eos_pad_value = 1.0
        self._label_pad_value = -100
        self.augment = True

        self.imgs = [
            read_image(str(p))
            for p in tqdm(
                sorted(folder_path.glob("*.png"), key=lambda s: float(re.search(r"\d+", str(s))[0])),
                desc="Loading images",
                unit="files"
            )
        ]

        with (folder_path / "digitStruct.json").open() as f:
            bbox_label_data = json.load(f)

        self.labels = [
            torch.tensor([bbox_lbl_item['label'] % 10 for bbox_lbl_item in bbox_lbl_items], dtype=torch.long)
            for bbox_lbl_items in tqdm(bbox_label_data, desc="Loading labels")
        ]

        self.oes = []
        for lbls in tqdm(self.labels, desc="Building eos vectors"):
            eos = torch.zeros_like(lbls, dtype=torch.float32)
            eos[-1] = 1.0
            self.oes.append(eos)

        self.bboxes = [
            BoundingBoxes(
                [
                    bbox_lbl['bbox']
                    for bbox_lbl in bbox_lbl_items
                ],
                format=BoundingBoxFormat.XYXY,
                canvas_size=img.shape[-2:]
            )
            for bbox_lbl_items, img in tqdm(zip(bbox_label_data, self.imgs), desc="Loading bboxes")
        ]

        preprocess_transforms = Compose([
            Resize(size=img_size),
            ToDtype(torch.float32, scale=True)
        ])

        self.imgs, self.bboxes = zip(*[
            preprocess_transforms(img, bbox)
            for img, bbox in tqdm(zip(self.imgs, self.bboxes), desc="Preprossessing imgs and bboxes")
        ])

        self.augmentation_transforms = Compose([
            RandomPerspective(perspective_dist, augment_prob),
            RandomApply([ColorJitter(brightness=cj_brightness, contrast=cj_contrast, saturation=cj_saturation)], augment_prob),
            RandomApply([RandomAffine(degrees=aff_deg, translate=aff_trans, scale=aff_scale, shear=aff_shear)], augment_prob),
            RandomApply([GaussianBlur(blur_kernel, sigma=blur_sigma)], augment_prob),
        ])

    def __getitem__(self, item):

        if self.augment:
            img, bboxes = self.augmentation_transforms(self.imgs[item], self.bboxes[item])
            bboxes = bboxes[torch.randperm(bboxes.shape[0])]
        else:
            img = self.imgs[item]
            bboxes = self.bboxes[item]
        per_channel_means = img.mean(dim=[1, 2])
        per_channel_stds = img.std(dim=[1, 2])
        img = Normalize(mean=per_channel_means, std=per_channel_stds)(img)
        normalized_center = (bboxes[:, :2] + (bboxes[:, 2:4] / 2)) / torch.tensor([[img.shape[2], img.shape[1]]])

        return img, normalized_center, self.labels[item], self.oes[item], per_channel_means, per_channel_stds

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        batched_imgs = torch.stack([b[0] for b in batch])
        batched_bboxes = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=self._bbox_pad_value)
        batched_lbls = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=self._label_pad_value)
        batched_eos = pad_sequence([b[3] for b in batch], batch_first=True, padding_value=self._eos_pad_value)
        batched_means = torch.stack([b[4] for b in batch])
        batched_stds = torch.stack([b[5] for b in batch])

        return batched_imgs, batched_bboxes, batched_lbls, batched_eos, batched_means, batched_stds


