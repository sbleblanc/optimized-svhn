from typing import Any, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet152, resnet50
from torchvision.transforms.v2.functional import to_dtype
from torchvision.ops import box_convert, box_iou
from torchvision.utils import draw_bounding_boxes
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
        self.cnn = resnet50(num_classes=4096)
        self.lstm = nn.LSTM(4096, 4096, 8, batch_first=True, dropout=0.05, proj_size=16)
        self.bbox_criterion = nn.MSELoss()
        self.num_eos_criterion = nn.CrossEntropyLoss()

        self.lambda_bbox = 1.0
        self.lambda_num = 1.0
        self.lambda_eos = 1.0

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
        imgs, bboxes, lbls, eos, means, stds = batch
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
        imgs, bboxes, lbls, eos, means, stds = batch
        bbox_loss, num_loss, eos_loss, bbox_out, num_out, eos_out = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * bbox_loss + self.lambda_num * num_loss + self.lambda_eos * eos_loss

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            self.validation_metrics['loss'].append(loss)
            self.validation_metrics['bbox_loss'].append(bbox_loss)
            self.validation_metrics['num_loss'].append(num_loss)
            self.validation_metrics['eos_loss'].append(eos_loss)

            ious = box_iou(bbox_out, bboxes.reshape(-1, 4))
            self.validation_metrics['iou'].append(ious.max(dim=1)[0].mean())

            num_preds = num_out.reshape((bboxes.shape[0], -1, 10)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            eos_preds = eos_out.reshape((bboxes.shape[0], -1, 2)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            num_truth = lbls.reshape(-1)[eos.reshape(-1) != -100]
            eos_truth = eos.reshape(-1)[eos.reshape(-1) != -100]

            self.validation_metrics['num_acc'].append(num_preds == num_truth)
            self.validation_metrics['eos_acc'].append(eos_preds == eos_truth)

            if batch_idx == 0:
                pred_scaled_bboxes = bbox_out.detach().clone()
                pred_scaled_bboxes[:, [0, 2]] *= imgs[0].shape[2]
                pred_scaled_bboxes[:, [1, 3]] *= imgs[0].shape[1]
                pred_scaled_bboxes = pred_scaled_bboxes.reshape(bboxes.shape[0], -1, 4)
                pred_box_count = eos_out.cpu().argmax(dim=-1).reshape(eos.shape[0], -1).argmax(dim=1)
                pred_lbls = num_out.cpu().argmax(dim=-1).reshape(lbls.shape[0], -1)

                true_scaled_bboxes = bboxes.reshape(-1, 4).detach().clone()
                true_scaled_bboxes[:, [0, 2]] *= imgs[0].shape[2]
                true_scaled_bboxes[:, [1, 3]] *= imgs[0].shape[1]
                true_scaled_bboxes = true_scaled_bboxes.reshape(bboxes.shape[0], -1, 4)
                expanded_means = means.unsqueeze(-1).unsqueeze(-1)
                expanded_stds = stds.unsqueeze(-1).unsqueeze(-1)
                unnorm_imgs = imgs * expanded_stds + expanded_means
                true_box_count = eos.argmax(dim=1).cpu()
                for i in range(imgs.shape[0]):
                    num_true_labels = true_box_count[i] + 1
                    true_boxes = true_scaled_bboxes[i, :num_true_labels]
                    true_labels = [str(l.item()) for l in lbls[i, :num_true_labels]]
                    num_pred_labels = pred_box_count[i] + 1
                    pred_boxes = pred_scaled_bboxes[i, :num_pred_labels]
                    pred_labels = [str(l.item()) for l in pred_lbls[i, :num_pred_labels]]
                    colors = ["blue"] * num_true_labels + ["green"] * num_pred_labels
                    img_with_box = draw_bounding_boxes(
                        image=to_dtype(unnorm_imgs[i], torch.uint8, scale=True),
                        boxes=torch.cat([true_boxes, torch.clamp(pred_boxes, min=0)], dim=0),
                        labels=true_labels + pred_labels,
                        colors=colors
                    )
                    self.logger.experiment.log_image(
                        run_id=self.logger.run_id,
                        image=img_with_box.permute(1, 2, 0).cpu().detach().numpy(),
                        artifact_file=f"bboxes/epoch_{self.current_epoch}/{i}_{''.join(true_labels)}.png"
                    )





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
