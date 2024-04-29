import numpy as np
from typing import Sequence, Tuple
from tqdm import tqdm
from torchvision.io import read_image, ImageReadMode


def get_avg_img_size(img_paths: Sequence[str], mode: ImageReadMode) -> Tuple[float, float]:
    shapes = [
        read_image(p, mode).shape[1:]
        for p in tqdm(img_paths)
    ]
    heights = np.array([s[0] for s in shapes])
    widths = np.array([s[1] for s in shapes])
    return heights.mean(), widths.mean()
