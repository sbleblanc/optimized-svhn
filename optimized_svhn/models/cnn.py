from typing import Any, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import get_model
from torchvision.transforms.v2.functional import to_dtype
from torchvision.ops import box_iou, generalized_box_iou_loss, generalized_box_iou
from torchvision.utils import draw_bounding_boxes, draw_keypoints
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
            cnn_out_dropout: float,
            max_length: int
    ):
        super().__init__()

        self.cnn = get_model(cnn_model, num_classes=cnn_out_dim * max_length)
        self.cnn_dropout = nn.Dropout(p=cnn_out_dropout)
        self.lstm = nn.LSTM(cnn_out_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=lstm_dropout, proj_size=3)
        self.bbox_criterion = nn.MSELoss()
        self.inverse_criterion = nn.MSELoss()
        self.num_eos_criterion = nn.CrossEntropyLoss()
        self.eos_criterion = nn.BCEWithLogitsLoss()

        self.lambda_bbox = bbox_lambda
        self.lambda_num = lbl_lambda
        self.lambda_eos = eos_lambda
        self.learning_rate = learning_rate
        self.max_length = max_length

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

    def compute_loss(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, bboxes, lbls, eos, means, stds = batch
        cnn_out = self.cnn_dropout(self.cnn(imgs))
        # lstm_input = cnn_out.expand(lbls.shape[1], -1, -1).permute((1, 0, 2))
        lstm_input = cnn_out.reshape(imgs.shape[0], self.max_length, -1)[:, :bboxes.shape[1]]
        lstm_output, _ = self.lstm(lstm_input)

        # num_pred_coord = (torch.sigmoid(lstm_output[:, :, -1]) > 0.5).long().argmax(-1) + 1
        # pred_coord = torch.cat([
        #     lstm_output[i, :n, :2].reshape(-1, 2)
        #     for i, n in enumerate(num_pred_coord)
        # ])
        # truth_coord = torch.cat([
        #     bboxes[i, :n].reshape(-1, 2)
        #     for i, n in enumerate(num_pred_coord)
        # ])

        lstm_output = lstm_output.reshape(-1, 3)
        coord_out = lstm_output[:, :2]
        eos_out = lstm_output[:, 2:3]

        coord_loss = self.bbox_criterion(coord_out, bboxes.reshape(-1, 2))
        eos_loss = self.eos_criterion(eos_out, eos.reshape(-1, 1))

        return coord_loss, eos_loss, coord_out, eos_out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        coord_loss, eos_loss, _, _ = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * coord_loss + self.lambda_eos * eos_loss

        self.log_dict(
            {
                "loss": loss,
                "coord_loss": coord_loss,
                "eos_loss": eos_loss,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, bboxes, lbls, eos, means, stds = batch
        coord_loss, eos_loss, coord_out, eos_out = self.compute_loss(batch, batch_idx)
        loss = self.lambda_bbox * coord_loss + self.lambda_eos * eos_loss

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:

            eos_preds = (torch.sigmoid(eos_out) > 0.5).long().squeeze()
            eos_truth = eos.reshape(-1)[eos.reshape(-1) != -100]

            self.log_dict(
                {
                    "val_loss": loss,
                    "val_coord_loss": coord_loss,
                    "val_eos_loss": eos_loss,
                    "val_eos_acc": (eos_preds == eos_truth).float().mean()
                },
                prog_bar=True,
                on_epoch=True,
                on_step=False
            )

        if batch_idx == 0:
            pred_scaled_coord = coord_out.detach().cpu()
            pred_scaled_coord = pred_scaled_coord * torch.tensor([[imgs[0].shape[2], imgs[0].shape[1]]])
            pred_scaled_coord = pred_scaled_coord.reshape(bboxes.shape[0], -1, 2)
            pred_coord_count = (torch.sigmoid(eos_out.detach().cpu().reshape(eos.shape[0], -1)) > 0.5).long().argmax(dim=-1) + 1

            true_scaled_coord = bboxes.reshape(-1, 2).detach().cpu()
            true_scaled_coord = true_scaled_coord * torch.tensor([[imgs[0].shape[2], imgs[0].shape[1]]])
            true_scaled_coord = true_scaled_coord.reshape(bboxes.shape[0], -1, 2)
            true_coord_count = eos.argmax(dim=1).cpu() + 1

            expanded_means = means.unsqueeze(-1).unsqueeze(-1)
            expanded_stds = stds.unsqueeze(-1).unsqueeze(-1)
            unnorm_imgs = imgs * expanded_stds + expanded_means

            for i in range(imgs.shape[0]):
                pred_coord = pred_scaled_coord[i, :pred_coord_count[i]].unsqueeze(1)
                true_coord = true_scaled_coord[i, :true_coord_count[i]].unsqueeze(1)
                img_with_key = draw_keypoints(to_dtype(unnorm_imgs[i], torch.uint8, scale=True), true_coord, colors="blue")
                img_with_key = draw_keypoints(to_dtype(img_with_key, torch.uint8, scale=True), pred_coord, colors="red")
                self.logger.experiment.log_image(
                    run_id=self.logger.run_id,
                    image=img_with_key.permute(1, 2, 0).cpu().detach().numpy(),
                    artifact_file=f"bboxes/epoch_{self.current_epoch}/{i}.png"
                )

                # pred_scaled_bboxes = bbox_out.detach().clone()
                # pred_scaled_bboxes[:, [0, 2]] *= imgs[0].shape[2]
                # pred_scaled_bboxes[:, [1, 3]] *= imgs[0].shape[1]
                # pred_scaled_bboxes = pred_scaled_bboxes.reshape(bboxes.shape[0], -1, 4)
                # pred_box_count = eos_out.cpu().argmax(dim=-1).reshape(eos.shape[0], -1).argmax(dim=1)
                # pred_lbls = num_out.cpu().argmax(dim=-1).reshape(lbls.shape[0], -1)
                #
                # true_scaled_bboxes = bboxes.reshape(-1, 4).detach().clone()
                # true_scaled_bboxes[:, [0, 2]] *= imgs[0].shape[2]
                # true_scaled_bboxes[:, [1, 3]] *= imgs[0].shape[1]
                # true_scaled_bboxes = true_scaled_bboxes.reshape(bboxes.shape[0], -1, 4)
                # expanded_means = means.unsqueeze(-1).unsqueeze(-1)
                # expanded_stds = stds.unsqueeze(-1).unsqueeze(-1)
                # unnorm_imgs = imgs * expanded_stds + expanded_means
                # true_box_count = eos.argmax(dim=1).cpu()
                # for i in range(imgs.shape[0]):
                #     num_true_labels = true_box_count[i] + 1
                #     true_boxes = true_scaled_bboxes[i, :num_true_labels]
                #     true_labels = [str(l.item()) for l in lbls[i, :num_true_labels]]
                #     num_pred_labels = pred_box_count[i] + 1
                #     pred_boxes = pred_scaled_bboxes[i, :num_pred_labels]
                #     pred_labels = [str(l.item()) for l in pred_lbls[i, :num_pred_labels]]
                #     colors = ["blue"] * num_true_labels + ["green"] * num_pred_labels
                #     try:
                #         img_with_box = draw_bounding_boxes(
                #             image=to_dtype(unnorm_imgs[i], torch.uint8, scale=True),
                #             boxes=torch.cat([true_boxes, torch.clamp(pred_boxes, min=0)], dim=0),
                #             labels=true_labels + pred_labels,
                #             colors=colors
                #         )
                #         self.logger.experiment.log_image(
                #             run_id=self.logger.run_id,
                #             image=img_with_box.permute(1, 2, 0).cpu().detach().numpy(),
                #             artifact_file=f"bboxes/epoch_{self.current_epoch}/{i}_{''.join(true_labels)}.png"
                #         )
                #     except:
                #         print(pred_boxes)

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
