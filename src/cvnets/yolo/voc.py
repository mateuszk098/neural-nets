import os
from collections import namedtuple
from enum import StrEnum
from os import PathLike
from pathlib import Path
from xml.etree import ElementTree

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


Detection = namedtuple("Detection", ("label", "bbox"))
ImageMeta = namedtuple("ImageMeta", ("path", "width", "height", "detections"))


class VOCSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TRAINVAL = "trainval"
    TEST = "test"


def load_voc_dataset(root: str | PathLike, split: VOCSplit) -> tuple[list[ImageMeta], list[str]]:
    data_root = Path(root).expanduser()
    split = VOCSplit(split)

    dataset = list()
    classes = set()

    for voc in data_root.iterdir():
        if split == VOCSplit.TEST and voc.name != "VOC2007":
            continue

        image_set = voc.joinpath("ImageSets", "Main", f"{split}.txt")

        with open(image_set) as f:
            for name in (name.strip() for name in f):
                image_path = voc.joinpath("JPEGImages", f"{name}.jpg")
                annot_path = voc.joinpath("Annotations", f"{name}.xml")

                annot_tree = ElementTree.parse(annot_path)
                annot_root = annot_tree.getroot()

                detections = list()
                # Following elements are always present, so don't mind about the type ignore.
                for obj in annot_root.findall("object"):
                    if obj.find("difficult").text == "1":  # type: ignore
                        continue

                    label = obj.find("name").text  # type: ignore
                    bndbox = obj.find("bndbox")  # type: ignore
                    bbox = (
                        int(bndbox.find("xmin").text),  # type: ignore
                        int(bndbox.find("ymin").text),  # type: ignore
                        int(bndbox.find("xmax").text),  # type: ignore
                        int(bndbox.find("ymax").text),  # type: ignore
                    )
                    detections.append(Detection(label, bbox))
                    classes.add(label)

                if detections:
                    width = int(annot_root.find("size").find("width").text)  # type: ignore
                    height = int(annot_root.find("size").find("height").text)  # type: ignore
                    dataset.append(ImageMeta(image_path, width, height, detections))

    return dataset, sorted(classes)
