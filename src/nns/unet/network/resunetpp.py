import torch
import torch.nn as nn
from torch.types import Tensor


class ResUNetPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base)
        self.decoder = Decoder(out_channels, base)

    def forward(self, x: Tensor) -> Tensor:
        eb1, eb2, eb3, eb4 = self.encoder(x)
        return self.decoder(eb1, eb2, eb3, eb4)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base: int) -> None:
        super().__init__()
        self.eb1 = InputBlock(in_channels, base, kernel_size=3)
        self.se1 = SqueezeExcitationBlock(base)

        self.eb2 = ResidualBlock(base, base * 2, kernel_size=3, stride=2)
        self.se2 = SqueezeExcitationBlock(base * 2)

        self.eb3 = ResidualBlock(base * 2, base * 4, kernel_size=3, stride=2)
        self.se3 = SqueezeExcitationBlock(base * 4)

        self.eb4 = ResidualBlock(base * 4, base * 8, kernel_size=3, stride=2)
        self.se4 = SqueezeExcitationBlock(base * 8)

        self.aspp = ASPPBlock(base * 8, base * 16)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        eb1 = self.eb1(x)
        eb1 = self.se1(eb1)

        eb2 = self.eb2(eb1)
        eb2 = self.se2(eb2)

        eb3 = self.eb3(eb2)
        eb3 = self.se3(eb3)

        eb4 = self.eb4(eb3)
        eb4 = self.se4(eb4)
        eb4 = self.aspp(eb4)

        return eb1, eb2, eb3, eb4


class Decoder(nn.Module):
    def __init__(self, out_channels: int, base: int) -> None:
        super().__init__()
        self.db1 = DecodingBlock((base * 4, base * 16), base * 8)
        self.db2 = DecodingBlock((base * 2, base * 8), base * 4)
        self.db3 = DecodingBlock((base, base * 4), base * 2)

        self.aspp = ASPPBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)

    def forward(self, eb1: Tensor, eb2: Tensor, eb3: Tensor, eb4: Tensor) -> Tensor:
        db1 = self.db1(eb3, eb4)
        db2 = self.db2(eb2, db1)
        db3 = self.db3(eb1, db2)

        aspp = self.aspp(db3)

        return self.out(aspp)


class InputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, factor: int = 8) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // factor),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // factor, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        se = self.se(x)
        return x * se.view(x.size(0), x.size(1), 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ASPPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.aspp = nn.ModuleList()
        for rate in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [block(x) for block in self.aspp]
        return self.fusion(torch.cat(features, dim=1))


class DecodingBlock(nn.Module):
    def __init__(self, in_channels: tuple[int, int], out_channels: int) -> None:
        super().__init__()
        self.at = AttentionBlock(in_channels[0], in_channels[1])
        self.up = UpsamplingBlock(in_channels[1], out_channels)
        self.db = ResidualBlock(in_channels[0] + out_channels, out_channels, kernel_size=3)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        at = self.at(x1, x2)
        up = self.up(at)
        return self.db(torch.cat((x1, up), dim=1))


class AttentionBlock(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int) -> None:
        super().__init__()
        self.at1 = nn.Sequential(
            nn.BatchNorm2d(in_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels1, in_channels2, kernel_size=3, padding=1, stride=2),
        )
        self.at2 = nn.Sequential(
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1),
        )
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x2 * self.fusion(self.at1(x1) + self.at2(x2))


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
