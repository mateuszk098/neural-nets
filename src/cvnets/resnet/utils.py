from torchvision import transforms
from torchvision.datasets import ImageFolder


class TransformMixin:
    _train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(contrast=(0.8, 1.2), saturation=(0.8, 1.2), brightness=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.25),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    _eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def train(self) -> None:
        self.transform = self._train_transform

    def eval(self) -> None:
        self.transform = self._eval_transform


class MyImageFolder(ImageFolder, TransformMixin): ...
