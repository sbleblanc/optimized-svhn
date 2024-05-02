from typing import Any, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import get_model
from torchvision.transforms.v2.functional import to_dtype
from torchvision.ops import box_iou, generalized_box_iou_loss, generalized_box_iou
from torchvision.utils import draw_bounding_boxes
from lightning import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class SVHNModel(LightningModule):

    def __init__(
            self,
            bbox_lambda: float,
            lbl_lambda: float,
            eos_lambda: float,
            cnn_model: str,
            cnn_out_dim: int,
            num_lstm_layers: int,
            lstm_hidden_dim: int,
            lstm_dropout: float,
            learning_rate: float,
            cnn_out_dropout: float
    ):
        super().__init__()

        self.cnn = get_model(cnn_model, num_classes=cnn_out_dim)
        self.cnn_dropout = nn.Dropout(p=cnn_out_dropout)
        self.lstm = nn.LSTM(cnn_out_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=lstm_dropout, proj_size=16)
        self.bbox_criterion = nn.MSELoss()
        self.inverse_criterion = nn.MSELoss()
        self.num_eos_criterion = nn.CrossEntropyLoss()

        self.lambda_bbox = bbox_lambda
        self.lambda_num = lbl_lambda
        self.lambda_eos = eos_lambda
        self.learning_rate = learning_rate

        # self.validation_metrics = {
        #     "loss": [],
        #     "bbox_loss": [],
        #     "num_loss": [],
        #     "eos_loss": [],
        #     "iou": [],
        #     "num_acc": [],
        #     "eos_acc": [],
        # }

        self.__pause_logging = False

    def compute_loss(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, bboxes, lbls, eos, means, stds = batch
        cnn_out = self.cnn_dropout(self.cnn(imgs))
        lstm_input = cnn_out.expand(lbls.shape[1], -1, -1).permute((1, 0, 2))
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output.reshape(-1, 16)
        bbox_out = lstm_output[:, :4]
        num_class_out = lstm_output[:, 4:14]
        eos_out = lstm_output[:, 14:16]

        # inverse_target = torch.zeros(bbox_out.shape[0], device=bbox_out.device)
        # x_inverse_loss = self.inverse_criterion(
        #     torch.minimum(bbox_out[:, 2] - bbox_out[:, 0], inverse_target),
        #     inverse_target
        # )
        # y_inverse_loss = self.inverse_criterion(
        #     torch.minimum(bbox_out[:, 3] - bbox_out[:, 1], inverse_target),
        #     inverse_target
        # )
        # inverse_loss = x_inverse_loss + y_inverse_loss
        #
        # ious = []
        # for num_boxes, boxes in zip(eos_out.argmax(dim=-1).reshape(eos.shape[0], -1).argmax(dim=1) + 1, bbox_out.reshape(bboxes.shape[0], -1, 4)):
        #     # same_box_mask = 1 - torch.eye(num_boxes, device=boxes.device)
        #     expected_boxes = boxes[:num_boxes]
        #     iou = box_iou(expected_boxes, expected_boxes).sum() - num_boxes
        #     print(iou)
        #     print(num_boxes)
        #     # iou = 2 - generalized_box_iou_loss(expected_boxes, expected_boxes)
        #     # iou = box_iou(expected_boxes, expected_boxes) * same_box_mask
        #     ious.append(iou)
        #
        # iou_loss = torch.stack(ious).mean()

        bbox_loss = self.bbox_criterion(bbox_out, bboxes.reshape(-1, 4))
        num_loss = self.num_eos_criterion(num_class_out, lbls.reshape(-1, 1).squeeze())
        eos_loss = self.num_eos_criterion(eos_out, eos.reshape(-1, 1).squeeze())

        return bbox_loss, num_loss, eos_loss, bbox_out, num_class_out, eos_out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        bbox_loss, num_loss, eos_loss, _, _, _ = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * bbox_loss + self.lambda_num * num_loss + self.lambda_eos * eos_loss
        # loss = num_loss

        # self.logger.log_metrics({
        #     "loss": loss.item(),
        #     "bbox_loss": bbox_loss.item(),
        #     "num_loss": num_loss.item(),
        #     "eos_loss": eos_loss.item()
        # }, step=self.global_step)

        self.log_dict(
            {
                "loss": loss,
                "bbox_loss": bbox_loss,
                "num_loss": num_loss,
                "eos_loss": eos_loss,
                # "inverse_loss": inverse_loss,
                # "iou_loss": iou_loss
            },
            prog_bar=True,
            on_epoch=True,
            on_step=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, bboxes, lbls, eos, means, stds = batch
        bbox_loss, num_loss, eos_loss, bbox_out, num_out, eos_out = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * bbox_loss + self.lambda_num * num_loss + self.lambda_eos * eos_loss

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            # self.validation_metrics['loss'].append(loss)
            # self.validation_metrics['bbox_loss'].append(bbox_loss)
            # self.validation_metrics['num_loss'].append(num_loss)
            # self.validation_metrics['eos_loss'].append(eos_loss)

            ious = box_iou(bbox_out, bboxes.reshape(-1, 4))
            # self.validation_metrics['iou'].append(ious.max(dim=1)[0].mean())

            num_preds = num_out.reshape((bboxes.shape[0], -1, 10)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            eos_preds = eos_out.reshape((bboxes.shape[0], -1, 2)).argmax(dim=-1).reshape(-1)[eos.reshape(-1) != -100]
            num_truth = lbls.reshape(-1)[eos.reshape(-1) != -100]
            eos_truth = eos.reshape(-1)[eos.reshape(-1) != -100]

            self.log_dict(
                {
                    "val_loss": loss,
                    "val_bbox_loss": bbox_loss,
                    "val_num_loss": num_loss,
                    "val_eos_loss": eos_loss,
                    # "val_inverse_loss": inverse_loss,
                    # "val_iou_loss": iou_loss,
                    "val_iou": ious.max(dim=1)[0].mean(),
                    "val_num_acc": (num_preds == num_truth).float().mean(),
                    "val_eos_acc": (eos_preds == eos_truth).float().mean()
                },
                prog_bar=True,
                on_epoch=True,
                on_step=False
            )

            # self.validation_metrics['num_acc'].append(num_preds == num_truth)
            # self.validation_metrics['eos_acc'].append(eos_preds == eos_truth)

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
                    try:
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
                    except:
                        print(pred_boxes)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def on_validation_epoch_start(self) -> None:
    #     self.validation_metrics['loss'].clear()
    #     self.validation_metrics['bbox_loss'].clear()
    #     self.validation_metrics['num_loss'].clear()
    #     self.validation_metrics['eos_loss'].clear()
    #     self.validation_metrics['iou'].clear()
    #     self.validation_metrics['num_acc'].clear()
    #     self.validation_metrics['eos_acc'].clear()

    # def on_validation_epoch_end(self) -> None:
    #
    #     if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
    #         self.logger.log_metrics(
    #             {
    #                 f"val_{m}": torch.stack(self.validation_metrics[m]).mean().item()
    #                 for m in ['loss', 'bbox_loss', 'num_loss', 'eos_loss', 'iou']
    #             },
    #             step=self.current_epoch
    #         )
    #         self.logger.log_metrics(
    #             {
    #                 f"val_{m}": torch.cat(self.validation_metrics[m]).float().mean().item()
    #                 for m in ['num_acc', 'eos_acc']
    #             },
    #             step=self.current_epoch
    #         )
    #         self.log("val_loss", torch.stack(self.validation_metrics["loss"]).mean().item(), on_epoch=True)
    #         self.log("val_num_acc", torch.cat(self.validation_metrics["num_acc"]).float().mean().item(), on_epoch=True)
