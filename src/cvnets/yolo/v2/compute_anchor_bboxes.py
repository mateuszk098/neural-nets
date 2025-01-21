import logging
from argparse import ArgumentParser
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric

from cvnets.yolo.voc import load_voc_dataset

plt.rcParams["text.usetex"] = True
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["font.family"] = "monospace"

plt.style.use("seaborn-v0_8-whitegrid")

CHECKPOINTS = Path("./checkpoints/")
CHECKPOINTS.mkdir(exist_ok=True)

FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def xyxy2xywh(bboxes: NDArray) -> NDArray:
    xywh = np.zeros_like(bboxes)
    xywh[:, :2] = bboxes[:, :2] + 0.5 * (bboxes[:, 2:] - bboxes[:, :2])
    xywh[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
    return xywh


def anchor_iou(whs1: NDArray, whs2: NDArray) -> NDArray:
    area1 = whs1[..., 0] * whs1[..., 1]
    area2 = whs2[..., 0] * whs2[..., 1]
    intersection = np.minimum(whs1[..., 0], whs2[..., 0]) * np.minimum(whs1[..., 1], whs2[..., 1])
    return intersection / (area1 + area2 - intersection + 1e-6)


def anchor_iou_distance(bbox: NDArray, cluster: NDArray) -> NDArray:
    iou_dist = 1 - anchor_iou(bbox.reshape(1, -1), cluster.reshape(1, -1))
    return iou_dist.item()


def calculate_mean_iou_per_cluster(whs_norm: NDArray, anchor_bboxes: NDArray, clusters_ids: list[list[int]]) -> None:
    ious = list()

    for cluster_ids, anchor_bbox in zip(clusters_ids, anchor_bboxes):
        n_elements = len(cluster_ids)
        cluster_ious = anchor_iou(whs_norm[cluster_ids], anchor_bbox)
        mean_cluster_iou = cluster_ious.mean()
        ious.append(cluster_ious)
        logging.info(f"Anchor BBox: {anchor_bbox} Elements: {n_elements} Mean IOU: {mean_cluster_iou:.4f}")

    ious = np.hstack(ious)
    logging.info(f"Overall Mean IOU: {ious.mean():.4f}")


def draw_anchor_bboxes(anchor_bboxes: NDArray, imgsz: int = 448, save: bool = False) -> NDArray:
    blank = np.full((imgsz, imgsz, 3), 0, dtype=np.uint8)
    for w, h in np.multiply(anchor_bboxes, imgsz):
        pt1 = (int(imgsz / 2) - int(w / 2), int(imgsz / 2) - int(h / 2))
        pt2 = (int(imgsz / 2) + int(w / 2), int(imgsz / 2) + int(h / 2))
        cv.rectangle(blank, pt1, pt2, (95, 255, 95), 2, cv.LINE_AA)
    if save:
        filename = CHECKPOINTS.joinpath(f"anchors_k_{len(anchor_bboxes)}.png")
        cv.imwrite(str(filename), cv.cvtColor(blank, cv.COLOR_RGB2BGR))
    return blank


def draw_whs_distribution(bboxes_wh_norm: NDArray, anchor_bboxes: NDArray) -> None:
    magnitude = np.hypot(bboxes_wh_norm[:, 0], bboxes_wh_norm[:, 1])
    plt.figure(figsize=(5, 5), tight_layout=True)
    plt.scatter(bboxes_wh_norm[:, 0], bboxes_wh_norm[:, 1], s=0.5, alpha=0.5, cmap="inferno", c=magnitude)
    plt.scatter(anchor_bboxes[:, 0], anchor_bboxes[:, 1], s=200, alpha=0.75, c="r", marker="X")
    plt.axis("equal")
    plt.xlabel(r"$W_{bbox}\big/W_{image}$")
    plt.ylabel(r"$H_{bbox}\big/H_{image}$")
    plt.title(f"Bounding Box WHs Distribution in VOC 2007/2012 - K = {len(anchor_bboxes)}")
    plt.savefig(CHECKPOINTS.joinpath(f"whs_distribution_k_{len(anchor_bboxes)}.png"))


def main(*, n_clusters: tuple[int, ...], dataset_path: Path) -> None:
    dataset, classes = load_voc_dataset(dataset_path, split="train")

    bboxes = list()
    imgwhs = list()

    for sample in dataset:
        for detection in sample.detections:
            bboxes.append(detection.bbox)
            imgwhs.append((sample.width, sample.height))

    bboxes = np.asarray(bboxes, dtype=np.float32)
    imgwhs = np.asarray(imgwhs, dtype=np.float32)
    bboxes_wh_norm = xyxy2xywh(bboxes)[:, 2:] / imgwhs

    for k in n_clusters:
        logging.info(f"Computing Anchor BBoxes for K = {k}...")

        initial_clusters = np.random.rand(k, 2)
        metric = distance_metric(type_metric.USER_DEFINED, func=anchor_iou_distance)

        km = kmeans(data=bboxes_wh_norm, initial_centers=initial_clusters, metric=metric)
        km = km.process()

        clusters_ids = km.get_clusters()
        anchor_bboxes = np.asarray(km.get_centers())
        np.save(CHECKPOINTS.joinpath(f"anchor_bboxes_k_{k}.npy"), anchor_bboxes)

        draw_whs_distribution(bboxes_wh_norm, anchor_bboxes)
        draw_anchor_bboxes(anchor_bboxes, save=True)
        calculate_mean_iou_per_cluster(bboxes_wh_norm, anchor_bboxes, clusters_ids)  # type: ignore


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--n-clusters", type=int, nargs="+", required=True)
    p.add_argument("--dataset-path", type=Path, required=False, default=Path("~/Documents/Datasets/VOC/"))
    main(**vars(p.parse_args()))
