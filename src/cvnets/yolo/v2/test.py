import random
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from cvnets.yolo.utils import load_yaml
from cvnets.yolo.v2.dataset import VOCDataset, collate_fn
from cvnets.yolo.v2.utils import load_anchor_bboxes

img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]


# def worker_init_fn(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset  # the dataset copy in this worker process
#     dataset.imgsz = random.choice(img_sizes)


if __name__ == "__main__":
    config_file = "config.yaml"
    config = SimpleNamespace(**load_yaml(config_file))
    anchor_bboxes = load_anchor_bboxes(config.ANCHOR_BBOXES)

    train_dataset = VOCDataset(
        config.DATASET,
        anchors=anchor_bboxes,
        downsample=config.DOWNSAMPLE,
        split="train",
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.manual_seed(42),
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    dataloader.dataset.train()
    print(dataloader.dataset.transform)
    for k, batch in enumerate(dataloader, start=1):
        pass  # print(dataloader.dataset.transform)

    dataloader.dataset.imgsz = random.choice(img_sizes)
    dataloader.dataset.eval()
    print(dataloader.dataset.transform)
    for k, batch in enumerate(dataloader, start=1):
        pass
    # for k, batch in enumerate(dataloader, start=1):
    #     print(dataloader.dataset.transform)
