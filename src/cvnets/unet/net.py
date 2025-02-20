import torch
import torch.nn as nn
from torch.types import Tensor


class UNet(nn.Module):
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
        self.eb1 = EncodingBlock(in_channels, base)
        self.mp1 = nn.MaxPool2d(2, 2)

        self.eb2 = EncodingBlock(base, base * 2)
        self.mp2 = nn.MaxPool2d(2, 2)

        self.eb3 = EncodingBlock(base * 2, base * 4)
        self.mp3 = nn.MaxPool2d(2, 2)

        self.eb4 = EncodingBlock(base * 4, base * 8)
        self.mp4 = nn.MaxPool2d(2, 2)

        self.eb5 = EncodingBlock(base * 8, base * 16)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        eb1 = self.eb1(x)
        mp1 = self.mp1(eb1)

        eb2 = self.eb2(mp1)
        mp2 = self.mp2(eb2)

        eb3 = self.eb3(mp2)
        mp3 = self.mp3(eb3)

        eb4 = self.eb4(mp3)
        mp4 = self.mp4(eb4)

        eb5 = self.eb5(mp4)

        return eb1, eb2, eb3, eb4, eb5


class Decoder(nn.Module):
    def __init__(self, out_channels: int, base: int) -> None:
        super().__init__()
        self.up1 = UpsamplingBlock(base * 16, base * 8)
        self.db1 = DecodingBlock(base * 16, base * 8)

        self.up2 = UpsamplingBlock(base * 8, base * 4)
        self.db2 = DecodingBlock(base * 8, base * 4)

        self.up3 = UpsamplingBlock(base * 4, base * 2)
        self.db3 = DecodingBlock(base * 4, base * 2)

        self.up4 = UpsamplingBlock(base * 2, base)
        self.db4 = DecodingBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)

    def forward(self, eb1: Tensor, eb2: Tensor, eb3: Tensor, eb4: Tensor, eb5: Tensor) -> Tensor:
        up1 = self.up1(eb5)  # 1024x1024 -> 512x512 (supposing base=64)
        db1 = self.db1(torch.cat((eb4, up1), dim=1))  # 2 * 512x512 -> 512x512

        up2 = self.up2(db1)  # 512x512 -> 256x256
        db2 = self.db2(torch.cat((eb3, up2), dim=1))  # 2 * 256x256 -> 256x256

        up3 = self.up3(db2)  # 256x256 -> 128x128
        db3 = self.db3(torch.cat((eb2, up3), dim=1))  # 2 * 128x128 -> 128x128

        up4 = self.up4(db3)  # 128x128 -> 64x64
        db4 = self.db4(torch.cat((eb1, up4), dim=1))  # 2 * 64x64 -> 64x64

        return self.out(db4)  # 64x64 -> 3x3


class EncodingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecodingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
