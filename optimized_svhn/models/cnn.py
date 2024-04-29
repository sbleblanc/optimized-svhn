from typing import Any, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet152
from torchvision.ops import box_convert, box_iou
from lightning import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class SVHNModel(LightningModule):

    def __init__(self):
        super().__init__()
        # conv1 = nn.Conv2d(1, 64, 5)
        # pool = nn.MaxPool2d(2, 2)
        # conv2 = nn.Conv2d(64, 128, 3)
        # fc1 = nn.Linear(128 * 13 * 28, 4096)
        # fc2 = nn.Linear(4096, 1024)
        #
        # self.cnn = nn.Sequential(
        #     conv1,
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     pool,
        #     conv2,
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     pool,
        #     nn.Flatten(),
        #     fc1,
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     fc2,
        #     nn.ReLU()
        # )

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 128, 7, 1),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(128, 256, 3),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(256, 128, 1),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Conv2d(128, 256, 3),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Conv2d(256, 256, 1),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Conv2d(256, 512, 3),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(512, 256, 1),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Conv2d(256, 1024, 3),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Flatten(),
        #     nn.Linear(1024 * 2 * 9, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.05)
        # )
        self.cnn = resnet152(num_classes=4096)
        self.lstm = nn.LSTM(4096, 4096, 8, batch_first=True, dropout=0.05, proj_size=16)
        self.bbox_criterion = nn.MSELoss()
        self.num_eos_criterion = nn.CrossEntropyLoss()

        self.lambda_bbox = 0.1
        self.lambda_num = 2.0
        self.lambda_eos = 2.0

        self.validation_metrics = {
            "loss": [],
            "bbox_loss": [],
            "num_loss": [],
            "eos_loss": [],
            "iou": [],
            "num_acc": [],
            "eos_acc": [],
        }

        self.__pause_logging = False

    def compute_loss(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, bboxes, lbls, eos = batch
        cnn_out = self.cnn(imgs)
        lstm_input = cnn_out.expand(lbls.shape[1], -1, -1).permute((1, 0, 2))
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output.reshape(-1, 16)
        bbox_out = lstm_output[:, :4]
        num_class_out = lstm_output[:, 4:14]
        eos_out = lstm_output[:, 14:16]

        bbox_loss = self.bbox_criterion(bbox_out, bboxes.reshape(-1, 4))
        num_loss = self.num_eos_criterion(num_class_out, lbls.reshape(-1, 1).squeeze())
        eos_loss = self.num_eos_criterion(eos_out, eos.reshape(-1, 1).squeeze())

        return bbox_loss, num_loss, eos_loss, bbox_out, num_class_out, eos_out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        bbox_loss, num_loss, eos_loss, _, _, _ = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * bbox_loss + self.lambda_num * num_loss + self.lambda_eos * eos_loss
        # loss = num_loss

        self.logger.log_metrics({
            "loss": loss.item(),
            "bbox_loss": bbox_loss.item(),
            "num_loss": num_loss.item(),
            "eos_loss": eos_loss.item()
        }, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, bboxes, lbls, eos = batch
        bbox_loss, num_loss, eos_loss, bbox_out, num_out, eos_out = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * bbox_loss + self.lambda_num * num_loss + self.lambda_eos * eos_loss
        # loss = num_loss

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            self.validation_metrics['loss'].append(loss)
            self.validation_metrics['bbox_loss'].append(bbox_loss)
            self.validation_metrics['num_loss'].append(num_loss)
            self.validation_metrics['eos_loss'].append(eos_loss)

            # truth_boxes = box_convert(
            #     bboxes.reshape(-1, 4),
            #     in_fmt="xywh",
            #     out_fmt="xyxy"
            # )
            #
            # pred_boxes = box_convert(
            #     bbox_out,
            #     in_fmt="xywh",
            #     out_fmt="xyxy"
            # )

            ious = box_iou(bbox_out, bboxes.reshape(-1, 4))
            self.validation_metrics['iou'].append(ious.max(dim=1)[0].mean())

            num_preds = num_out.reshape((bboxes.shape[0], -1, 10)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            eos_preds = eos_out.reshape((bboxes.shape[0], -1, 2)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            num_truth = lbls.reshape(-1)[eos.reshape(-1) != -100]
            eos_truth = eos.reshape(-1)[eos.reshape(-1) != -100]

            self.validation_metrics['num_acc'].append(num_preds == num_truth)
            self.validation_metrics['eos_acc'].append(eos_preds == eos_truth)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def on_validation_epoch_start(self) -> None:
        self.validation_metrics['loss'].clear()
        self.validation_metrics['bbox_loss'].clear()
        self.validation_metrics['num_loss'].clear()
        self.validation_metrics['eos_loss'].clear()
        self.validation_metrics['iou'].clear()
        self.validation_metrics['num_acc'].clear()
        self.validation_metrics['eos_acc'].clear()

    def on_validation_epoch_end(self) -> None:

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            self.logger.log_metrics(
                {
                    f"valid_{m}": torch.stack(self.validation_metrics[m]).mean().item()
                    for m in ['loss', 'bbox_loss', 'num_loss', 'eos_loss', 'iou']
                },
                step=self.current_epoch
            )
            self.logger.log_metrics(
                {
                    f"valid_{m}": torch.cat(self.validation_metrics[m]).float().mean().item()
                    for m in ['num_acc', 'eos_acc']
                },
                step=self.current_epoch
            )
            self.log("val_loss", torch.stack(self.validation_metrics["loss"]).mean().item())
