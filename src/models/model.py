import hydra
import lightning.pytorch as L
import torch
import torch.nn.functional as F
from topk.svm import SmoothTop1SVM
from torch import nn
from torch.optim import SGD, Adam
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)

from src.utils.metrics_utils import HorizontalPCC, HorizontalSCC


class ModelMultiBranchModule(L.LightningModule):
    def __init__(
        self,
        embed_dim=1536,
        stem_proj_dim=512,
        attn_proj_dim=256,
        hidden_head_dims=(),
        dropout=0,
        k_sample=8,
        num_classes=2,
        bag_weight=0.7,
        lr=1e-4,
        reg=1e-5,
        instance_delay_epochs=0,
        optim="none",
        lr_sched="none",
        classes=None,
        bag_loss="mse",
        mutual_exclusive="mutual_exclusive",
    ):
        pass


class ModelMultiBranchRModule(L.LightningModule):
    def __init__(
        self,
        network,
        bag_loss_fn,
        instance_loss_fn,
        optimizer,
        scheduler,
        classes,
        bag_weight=0.7,
    ):
        super().__init__()
        num_classes = len(classes)

        # self.save_hyperparameters(
        #     ignore=[
        #         "nn",
        #         "bag_loss_fn",
        #         "instance_loss_fn",
        #         "optimizer",
        #         "scheduler",
        #     ]
        # )

        self.model = network
        self.classes = classes
        self.bag_weight = bag_weight
        self.optimizer_factory = optimizer
        self.scheduler_cfg = scheduler

        self.bag_loss_fn = bag_loss_fn
        self.instance_loss_fn = instance_loss_fn

        # self.instance_loss_fn = nn.MultiMarginLoss()

        metrics = MetricCollection(
            {
                "bag": MetricCollection(
                    {
                        "sns": MulticlassRecall(
                            num_classes=num_classes,
                            average=None,
                        ),
                        "spc": MulticlassSpecificity(
                            num_classes=num_classes,
                            average=None,
                        ),
                        "acc": MulticlassAccuracy(
                            num_classes=num_classes,
                            average=None,
                        ),
                        # "auc": MulticlassAUROC(
                        #     num_classes=num_classes,
                        #     average=None,
                        # ),
                        # "prc": MulticlassPrecisionRecallCurve(
                        #     num_classes=num_classes,
                        #     average=None,
                        # ),
                        "mae": MeanAbsoluteError(
                            num_outputs=num_classes,
                        ),
                        "r2": R2Score(
                            multioutput="raw_values",
                        ),
                        "pcc_h": HorizontalPCC(),
                        "scc_h": HorizontalSCC(),
                        "pcc_v": PearsonCorrCoef(num_outputs=num_classes),
                        "scc_v": SpearmanCorrCoef(num_outputs=num_classes),
                        "loss": MeanMetric(),
                    }
                ),
                "inst": MetricCollection(
                    {
                        "acc": MulticlassAccuracy(num_classes=2),
                        "loss": MeanMetric(),
                    }
                ),
                "total_loss": MeanMetric(),
            }
        )

        self.metrics = MetricCollection(
            [metrics.clone(prefix=f"{stage}/") for stage in ["train", "val", "test"]]
        )

    def setup(self, stage: str):
        # ensure instance loss is on the correct device
        self.instance_loss_fn = self.instance_loss_fn.to(self.device)

    def forward(self, embeddings, bag_label, return_attention=False):
        return self.model(embeddings, bag_label, return_attention=return_attention)

    def stage_step(self, batch, batch_idx, stage):
        # get input
        embeddings = batch["embeddings"][0]  # remove batch dim
        bag_label = batch["bag_label"]
        bag_scores = batch["scores"]

        # forward pass
        results = self(embeddings, bag_label)

        # get results
        instance_logits, instance_targets = (
            results["instance_logits"],
            results["instance_targets"],
        )

        bag_logits, bag_target = (
            results["bag_logits"],
            results["bag_target"],
        )

        # bag loss
        bag_loss = self.bag_loss_fn(bag_logits, bag_scores)

        # instance loss
        instance_loss = 0.0
        for logits, targets in zip(instance_logits, instance_targets):
            instance_loss += self.instance_loss_fn(logits, targets)
        instance_loss /= len(instance_logits)

        # total loss
        total_loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * instance_loss

        # update metrics
        # update bag-level metrics
        self.update_metrics(
            stage,
            instance_logits,
            instance_targets,
            bag_logits,
            bag_target,
            bag_scores,
            instance_loss,
            bag_loss,
            total_loss,
        )

        return total_loss

    def update_metrics(
        self,
        stage,
        instance_logits,
        instance_targets,
        bag_logits,
        bag_target,
        bag_scores,
        instance_loss,
        bag_loss,
        total_loss,
    ):
        ## bag metrics
        # clf metrics
        bag_preds = bag_logits.argmax(dim=-1)

        self.metrics[f"{stage}/bag_sns"].update(bag_preds, bag_target)
        self.metrics[f"{stage}/bag_spc"].update(bag_preds, bag_target)
        self.metrics[f"{stage}/bag_acc"].update(bag_preds, bag_target)
        # self.metrics[f"{stage}/bag_auc"].update(bag_preds, bag_target)
        # self.metrics[f"{stage}/bag_prc"].update(bag_preds, bag_target)

        # rgr metrics
        self.metrics[f"{stage}/bag_mae"].update(bag_logits, bag_scores)
        self.metrics[f"{stage}/bag_r2"].update(bag_logits, bag_scores)
        self.metrics[f"{stage}/bag_pcc_h"].update(bag_logits, bag_scores)
        self.metrics[f"{stage}/bag_scc_h"].update(bag_logits, bag_scores)
        self.metrics[f"{stage}/bag_pcc_v"].update(bag_logits, bag_scores)
        self.metrics[f"{stage}/bag_scc_v"].update(bag_logits, bag_scores)

        # loss
        self.metrics[f"{stage}/bag_loss"].update(bag_loss)

        ## inst metrics

        # this can be split into per-class accuracy for better monitornig
        if torch.cat(instance_logits).numel():
            self.metrics[f"{stage}/inst_acc"].update(
                torch.cat(instance_logits), torch.cat(instance_targets)
            )
        self.metrics[f"{stage}/inst_loss"].update(instance_loss)

        self.metrics[f"{stage}/total_loss"].update(total_loss)

    def on_stage_epoch_end(self, stage):
        stage_metrics = {}

        for k, metric in self.metrics.items():
            # only log stage metrics
            if k.startswith(stage):
                # skip intsance metrics when not evaluating instances
                if "inst" in k:
                    stage_metrics[k] = metric.compute()
                elif all([m not in k for m in ["loss", "pcc_h", "scc_h"]]):
                    values = metric.compute()
                    for i, val in enumerate(values):
                        stage_metrics[k + f"_{self.classes[i]}"] = val
                    stage_metrics[k + "_avg"] = val.mean()
                else:
                    stage_metrics[k] = metric.compute()
                metric.reset()

        self.log_dict(stage_metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.stage_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.stage_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.stage_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        # get input
        embeddings = batch["embeddings"][0]  # remove batch dim
        bag_label = batch["bag_label"]

        # forward pass
        results = self(embeddings, bag_label, return_attention=True)

        return {
            "samples": batch["names"],
            "targets": batch["bag_label"],
            "scores": batch["scores"],
            "logits": results["bag_logits"],
            "tiles_logits": results["instance_logits"],
            "attention": results["attention"],
            "bag_embeddings": results["bag_embeddings"],
        }

    def on_train_epoch_end(self):
        self.on_stage_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_stage_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_stage_epoch_end("test")

    def on_train_epoch_start(self):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())

        if self.scheduler_cfg is None:
            return optimizer

        scheduler = hydra.utils.instantiate(
            self.scheduler_cfg,
            optimizer=optimizer,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
