import json
import click
from tqdm import tqdm
from typing import Tuple
from glob import glob
from pathlib import Path

import mat73
from mlflow.entities import Param
from torch.utils.data import random_split, DataLoader
from torchvision.io import ImageReadMode
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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
@click.option("--bbox-lambda", type=click.FLOAT, default=1.0)
@click.option("--lbl-lambda", type=click.FLOAT, default=1.0)
@click.option("--eos-lambda", type=click.FLOAT, default=1.0)
@click.option("--seed", type=click.INT, default=42)
@click.option("--experiment-name", type=click.STRING, default="svhn_cnn")
@click.option("--run-name", type=click.STRING, required=True)
@click.option("--train-split", type=click.FLOAT, default=0.8)
@click.option("--batch-size", type=click.INT, default=32)
@click.option("--cnn-model", type=click.STRING, default="resnet101")
@click.option("--cnn-out-dim", type=click.INT, default=4096)
@click.option("--num-lstm-layers", type=click.INT, default=4)
@click.option("--lstm-hidden-dim", type=click.INT, default=4096)
@click.option("--lstm-dropout", type=click.FLOAT, default=0.5)
@click.option("--img-size", type=click.Tuple([click.INT, click.INT]), default=(60, 120), nargs=2)
@click.option("--cj-brightness", type=click.FLOAT, default=0.5)
@click.option("--cj-contrast", type=click.FLOAT, default=0.5)
@click.option("--cj-saturation", type=click.FLOAT, default=0.5)
@click.option("--aff-deg", type=click.Tuple([click.INT, click.INT]), default=(-60, 60), nargs=2)
@click.option("--aff-trans", type=click.Tuple([click.FLOAT, click.FLOAT]), default=(0.5, 0.5), nargs=2)
@click.option("--aff-scale", type=click.Tuple([click.FLOAT, click.FLOAT]), default=(0.1, 3.0), nargs=2)
@click.option("--aff-shear", type=click.FLOAT, default=60)
@click.option("--blur-kernel", type=click.INT, default=5)
@click.option("--blur-sigma", type=click.Tuple([click.FLOAT, click.FLOAT]), default=(0.1, 1.5), nargs=2)
@click.option("--perspective-dist", type=click.FLOAT, default=0.7)
@click.option("--augment-prob", type=click.FLOAT, default=0.4)
@click.option("--learning-rate", type=click.FLOAT, default=1e-4)
@click.option("--gradient-clip", type=click.FLOAT, default=1.0)
@click.option("--precision", type=click.STRING, default="16-mixed")
@click.option("--cnn-out-dropout", type=click.FLOAT, default=0.5)
def train_model(
        train_set_folder: Path,
        bbox_lambda: float,
        lbl_lambda: float,
        eos_lambda: float,
        seed: int,
        experiment_name: str,
        run_name: str,
        train_split: float,
        batch_size: int,
        cnn_model: str,
        cnn_out_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_dropout: float,
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
        learning_rate: float,
        gradient_clip: float,
        precision: str,
        cnn_out_dropout: float
):
    seed_everything(seed, workers=True)

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name
    )

    params_dict = {
        "bbox_lambda": bbox_lambda,
        "lbl_lambda": lbl_lambda,
        "eos_lambda": eos_lambda,
        "seed": seed,
        "train_split": train_split,
        "batch_size": batch_size,
        "cnn_model": cnn_model,
        "cnn_out_dim": cnn_out_dim,
        "num_lstm_layers": num_lstm_layers,
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_dropout": lstm_dropout,
        "img_size": img_size,
        "cj_brightness": cj_brightness,
        "cj_contrast": cj_contrast,
        "cj_saturation": cj_saturation,
        "aff_deg": aff_deg,
        "aff_trans": aff_trans,
        "aff_scale": aff_scale,
        "aff_shear": aff_shear,
        "blur_kernel": blur_kernel,
        "blur_sigma": blur_sigma,
        "perspective_dist": perspective_dist,
        "augment_prob": augment_prob,
        "learning_rate": learning_rate,
        "gradient_clip": gradient_clip,
        "precision": precision,
        "cnn_out_dropout": cnn_out_dropout
    }

    mlflow_logger.experiment.log_batch(
        run_id=mlflow_logger.run_id,
        params=[Param(k, str(v)) for k, v in params_dict.items()]
    )

    model_callbacks = [
        ModelCheckpoint(
            f"models/{mlflow_logger.run_id}",
            save_top_k=1,
            filename="best_loss",
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=True
        ),
        # ModelCheckpoint(
        #     f"models/{mlflow_logger.run_id}",
        #     save_top_k=1,
        #     filename="best_num_acc",
        #     monitor="val_num_acc",
        #     mode="max",
        #     auto_insert_metric_name=True
        # ),
        ModelCheckpoint(
            f"models/{mlflow_logger.run_id}",
            filename="latest_epoch",
            every_n_epochs=1
        ),
        EarlyStopping(
            "val_loss",
            min_delta=1e-2,
            verbose=True,
            patience=5
        )
    ]

    ds = SVHNDataset(
        train_set_folder,
        img_size,
        cj_brightness,
        cj_contrast,
        cj_saturation,
        aff_deg,
        aff_trans,
        aff_scale,
        aff_shear,
        blur_kernel,
        blur_sigma,
        perspective_dist,
        augment_prob
    )
    train_ds, valid_ds = random_split(ds, [train_split, 1 - train_split])
    valid_ds.dataset.augment = False
    model = SVHNModel(
        bbox_lambda,
        lbl_lambda,
        eos_lambda,
        cnn_model,
        cnn_out_dim,
        num_lstm_layers,
        lstm_hidden_dim,
        lstm_dropout,
        learning_rate,
        cnn_out_dropout
    )

    trainer = Trainer(
        gradient_clip_val=gradient_clip,
        precision=precision,
        deterministic=True,
        logger=mlflow_logger,
        callbacks=model_callbacks,
        log_every_n_steps=200,
    )
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size=batch_size, collate_fn=ds.collate_fn, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size, collate_fn=ds.collate_fn, shuffle=False)
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
