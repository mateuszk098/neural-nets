# **Computer Vision Nets**

![Static Badge](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white&color=%234584b6)
![GitHub last commit](https://img.shields.io/github/last-commit/mateuszk098/computer-vision-nets?style=for-the-badge&color=%23fa9537)

Some neural network architectures implementation:

- ResNet (Deep Residual Learning for Image Recognition) &#8594; <https://arxiv.org/pdf/1512.03385>
- YOLOv1 (You Only Look Once: Unified, Real-Time Object Detection) &#8594; <https://arxiv.org/pdf/1506.02640>
- YOLOv2 (YOLO9000: Better, Faster, Stronger) &#8594; <https://arxiv.org/pdf/1612.08242>
- Variational Autoencoder (Auto-Encoding Variational Bayes) &#8594; <https://arxiv.org/pdf/1312.6114>
- UNet (U-Net: Convolutional Networks for Biomedical Image Segmentation) &#8594; <https://arxiv.org/pdf/1505.04597>
- ResUNet (Road Extraction by Deep Residual U-Net) &#8594; <https://arxiv.org/abs/1711.10684>

Build the project with uv &#8594; <https://github.com/astral-sh/uv>:

```bash
# If cuda is available.
uv sync --extra cu124
```

```bash
# If cuda is not available.
uv sync --extra cpu
```
