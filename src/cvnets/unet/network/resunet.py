import torch
import torch.nn as nn
from torch.types import Tensor


class ResUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 64) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base)
        self.decoder = Decoder(out_channels, base)

    def forward(self, x: Tensor) -> Tensor:
        eb1, eb2, eb3, eb4, eb5 = self.encoder(x)
        return self.decoder(eb1, eb2, eb3, eb4, eb5)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base: int) -> None:
        super().__init__()
        self.eb1 = InputResidualBlock(in_channels, base, kernel_size=3)
        self.eb2 = PreActivationResBlok(base, base * 2, kernel_size=3, stride=2)
        self.eb3 = PreActivationResBlok(base * 2, base * 4, kernel_size=3, stride=2)
        self.eb4 = PreActivationResBlok(base * 4, base * 8, kernel_size=3, stride=2)
        self.eb5 = PreActivationResBlok(base * 8, base * 16, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        eb1 = self.eb1(x)
        eb2 = self.eb2(eb1)
        eb3 = self.eb3(eb2)
        eb4 = self.eb4(eb3)
        eb5 = self.eb5(eb4)
        return eb1, eb2, eb3, eb4, eb5


class Decoder(nn.Module):
    def __init__(self, out_channels: int, base: int) -> None:
        super().__init__()
        self.up1 = UpsamplingBlock(base * 16, base * 8)
        self.db1 = PreActivationResBlok(base * 16, base * 8, kernel_size=3)

        self.up2 = UpsamplingBlock(base * 8, base * 4)
        self.db2 = PreActivationResBlok(base * 8, base * 4, kernel_size=3)

        self.up3 = UpsamplingBlock(base * 4, base * 2)
        self.db3 = PreActivationResBlok(base * 4, base * 2, kernel_size=3)

        self.up4 = UpsamplingBlock(base * 2, base)
        self.db4 = PreActivationResBlok(base * 2, base, kernel_size=3)

        self.out = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)

    def forward(self, eb1: Tensor, eb2: Tensor, eb3: Tensor, eb4: Tensor, eb5: Tensor) -> Tensor:
        up1 = self.up1(eb5)
        db1 = self.db1(torch.cat((eb4, up1), dim=1))

        up2 = self.up2(db1)
        db2 = self.db2(torch.cat((eb3, up2), dim=1))

        up3 = self.up3(db2)
        db3 = self.db3(torch.cat((eb2, up3), dim=1))

        up4 = self.up4(db3)
        db4 = self.db4(torch.cat((eb1, up4), dim=1))

        return self.out(db4)


class InputResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class PreActivationResBlok(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
