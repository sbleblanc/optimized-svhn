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
from torchvision.transforms.v2 import Resize, Compose, Normalize, ToDtype
import torchvision.transforms.functional as F
from torchvision.ops import box_convert


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class SVHNDataset(Dataset):

    def __init__(self, folder_path: Path):
        super().__init__()
        self._bbox_pad_value = 0.0
        self._eos_pad_value = -100
        self._label_pad_value = -100

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
            eos = torch.zeros_like(lbls, dtype=torch.long)
            eos[-1] = 1
            self.oes.append(eos)

        self.bboxes = [
            BoundingBoxes(
                [
                    box_convert(torch.tensor(bbox_lbl['bbox']), in_fmt="xywh", out_fmt="xyxy").tolist()
                    for bbox_lbl in bbox_lbl_items
                ],
                format=BoundingBoxFormat.XYXY,
                canvas_size=img.shape[-2:]
            )
            for bbox_lbl_items, img in tqdm(zip(bbox_label_data, self.imgs), desc="Loading bboxes")
        ]

        preprocess_transforms = Compose([
            Resize(size=(60, 120)),
            ToDtype(torch.float32, scale=True)
        ])

        self.imgs, self.bboxes = zip(*[
            preprocess_transforms(img, bbox)
            for img, bbox in tqdm(zip(self.imgs, self.bboxes), desc="Preprossessing imgs and bboxes")
        ])

        for b in self.bboxes:
            b[:, [0, 2]] /= 120.0
            b[:, [1, 3]] /= 60.0

        means = [img.mean(dim=[1, 2]) for img in self.imgs]
        stds = [img.std(dim=[1, 2]) for img in self.imgs]

        self.imgs = [
            Normalize(mean=m, std=s)(img)
            for img, m, s in tqdm(
                zip(self.imgs, means, stds),
                total=len(self.imgs),
                desc="Normalizing images",
                unit="imgs"
            )
        ]

    def __getitem__(self, item):
        return self.imgs[item], self.bboxes[item], self.labels[item], self.oes[item]

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        batched_imgs = torch.stack([b[0] for b in batch])
        batched_bboxes = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=self._bbox_pad_value)
        batched_lbls = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=self._label_pad_value)
        batched_eos = pad_sequence([b[3] for b in batch], batch_first=True, padding_value=self._eos_pad_value)

        return batched_imgs, batched_bboxes, batched_lbls, batched_eos


