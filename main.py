import json
import click
from tqdm import tqdm
from typing import Tuple
from glob import glob
from pathlib import Path

import mat73
from torch.utils.data import random_split, DataLoader
from torchvision.io import read_image, ImageReadMode
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from optimized_svhn.utils.img import get_avg_img_size
from optimized_svhn.data.loading import SVHNDataset
from optimized_svhn.utils.general import convert_to_list
from optimized_svhn.models.cnn import SVHNModel


@click.group()
def cli():
    pass


@cli.command()
@click.argument("train_set_folder", type=click.Path(file_okay=False, path_type=Path))
def train_model(train_set_folder: Path):
    seed_everything(42, workers=True)

    mlflow_logger = MLFlowLogger(
        experiment_name="svhn_cnn",
        run_name="testing"
    )

    model_callback = ModelCheckpoint(
        "models/resnet",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    ds = SVHNDataset(train_set_folder)
    train_ds, valid_ds = random_split(ds, [0.9, 0.1])
    model = SVHNModel()

    trainer = Trainer(
        gradient_clip_val=1.0,
        precision="16-mixed",
        deterministic=True,
        logger=mlflow_logger,
        callbacks=[model_callback]
    )
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size=32, collate_fn=ds.collate_fn),
        DataLoader(valid_ds, batch_size=32, collate_fn=ds.collate_fn)
    )


@cli.command()
@click.argument("mat_path", type=click.Path(dir_okay=False, path_type=Path))
def convert_mat_file_to_json(mat_path: Path):
    print("Loading .mat data...", end='')
    mat_dict = mat73.loadmat(mat_path)
    print("DONE")

    bbox_data = []

    for bbox_dict in tqdm(mat_dict['digitStruct']['bbox'], desc="Converting bboxes"):
        tops = convert_to_list(bbox_dict['top'])
        lefts = convert_to_list(bbox_dict['left'])
        widths = convert_to_list(bbox_dict['width'])
        heights = convert_to_list(bbox_dict['height'])
        labels = convert_to_list(bbox_dict['label'])

        bbox_data.append([
            {
                "bbox": [x, y, w, h],
                "label": l
            }
            for x, y, w, h, l in zip(lefts, tops, widths, heights, labels)
        ])

    with (mat_path.parent / f"{mat_path.stem}.json").open('w') as f:
        json.dump(bbox_data, f)




@cli.command()
@click.argument("image_paths", type=click.Path(dir_okay=False), nargs=-1)
def avg_img_sizes(image_paths: Tuple[str, ...]):
    complete_paths = [
        p
        for img_path in image_paths
        for p in glob(img_path)
    ]
    avg_h, avg_w = get_avg_img_size(complete_paths, ImageReadMode.GRAY)
    print(f"Average height: {avg_h}")
    print(f"Average width: {avg_w}")


if __name__ == "__main__":
    cli()
