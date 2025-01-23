import logging
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cvnets.yolo.utils import initialize_seed, load_yaml
from cvnets.yolo.v2.dataset import VOCDataset, collate_fn
from cvnets.yolo.v2.loss import NamedLoss, YOLOv2Loss
from cvnets.yolo.v2.net import YOLOv2
from cvnets.yolo.v2.utils import load_anchor_bboxes, postprocess_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

initialize_seed()
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metric: Metric,
    conf_thresh: float,
    nms_thresh: float,
) -> tuple[NamedLoss, dict[str, Any]]:
    model.eval()
    metric.reset()
    partial_loss = torch.zeros(5)

    with torch.inference_mode():
        for batch in loader:
            x, y = batch.images.to(DEVICE), batch.targets.to(DEVICE)
            predictions = model(x)

            loss = loss_fn(predictions, y)
            partial_loss += torch.as_tensor(loss)

            results = postprocess_predictions(
                predictions=predictions,
                anchors=loader.dataset.anchors,  # type: ignore
                downsample=loader.dataset.downsample,  # type: ignore
                imgsz=loader.dataset.imgsz,  # type: ignore
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
            )

            preds = list()
            target = list()

            # Iterate over the batch.
            for result in results:
                preds.append({"boxes": result.xyxys, "scores": result.confs, "labels": result.labels})

            for boxes, classes in zip(batch.bboxes, batch.labels):
                target.append({"boxes": boxes, "labels": classes})

            metric.update(preds, target)

    partial_loss = NamedLoss(*partial_loss.cpu().detach().div(len(loader)).tolist())
    mean_ap = metric.compute()

    return partial_loss, mean_ap


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    anchors = load_anchor_bboxes(config.ANCHOR_BBOXES)

    dataset = VOCDataset(
        config.DATASET,
        anchors=anchors,
        downsample=config.DOWNSAMPLE,
        imgsz=config.IMGSZ,
        split=config.EVAL_SPLIT,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    loader.dataset.eval()  # type: ignore

    logging.info("Loading model...")
    model = YOLOv2(num_anchors=dataset.num_anchors, num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(config.EVAL_CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], average="macro")
    loss_fn = YOLOv2Loss(lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)

    logging.info("Evaluating model...")
    loss, mean_ap = evaluate(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        metric=metric,
        conf_thresh=config.EVAL_CONF_THRESH,
        nms_thresh=config.EVAL_NMS_THRESH,
    )

    logging.info(f"{'Total Loss:':<25} {loss.total:.4f}")
    logging.info(f"{'Localization Loss:':<25} {loss.coord:.4f}")
    logging.info(f"{'Objectness Loss:':<25} {loss.itobj:.4f}")
    logging.info(f"{'Noobjectness Loss:':<25} {loss.noobj:.4f}")
    logging.info(f"{'Classification Loss:':<25} {loss.label:.4f}")
    logging.info(f"{'mAP@50:':<25} {mean_ap["map"]:.4f}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
