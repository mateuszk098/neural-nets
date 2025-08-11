import torch
from torch.types import Tensor
from torchvision.ops import nms

# def nms(detections: Tensor, iou_threshold: float = 0.3) -> Tensor:
#     sorted_dets = sorted(detections, key=lambda x: x[-1])  # type: ignore
#     keep = list()

#     while sorted_dets:
#         current_det = sorted_dets.pop()
#         sorted_dets = list(det for det in sorted_dets if iou(current_det, det[:-1]) < iou_threshold)
#         keep.append(current_det)

#     return torch.stack(keep, dim=0)


def decode_yolo_output(prediction: Tensor, imgsz: int, S: int, B: int) -> Tensor:
    device = prediction.device

    steps = torch.arange(S) / S
    offset_y, offset_x = torch.meshgrid(steps, steps, indexing="ij")
    offset_y = offset_y.reshape(-1, S, S, 1).repeat(1, 1, 1, B).to(device)
    offset_x = offset_x.reshape(-1, S, S, 1).repeat(1, 1, 1, B).to(device)

    pred_bboxes = prediction[..., : B * 5].reshape(-1, S, S, B, 5)
    # Extract the bounding box coordinates -> (batch_size, S, S, B, 4).
    xyxys = torch.stack(
        [
            # Predefined offsets are in 0 - 1, so we must normalize predicted offset by S
            # to be able to add it.
            pred_bboxes[..., 0] / S + offset_x - 0.5 * pred_bboxes[..., 2].square(),
            pred_bboxes[..., 1] / S + offset_y - 0.5 * pred_bboxes[..., 3].square(),
            pred_bboxes[..., 0] / S + offset_x + 0.5 * pred_bboxes[..., 2].square(),
            pred_bboxes[..., 1] / S + offset_y + 0.5 * pred_bboxes[..., 3].square(),
        ],
        dim=-1,
    )
    xyxys = xyxys.mul(imgsz)

    class_scores, class_ids = prediction[..., 5 * B :].max(dim=-1, keepdim=True)
    class_scores = class_scores.reshape(-1, S, S, 1, 1).repeat(1, 1, 1, B, 1)
    class_ids = class_ids.reshape(-1, S, S, 1, 1).repeat(1, 1, 1, B, 1)

    confs = pred_bboxes[..., 4].reshape(-1, S, S, B, 1)
    confs = confs * class_scores

    result = torch.cat((xyxys, confs, class_ids), dim=-1)
    return result.reshape(-1, S * S * B, 6)


def filter_detections(
    xyxys: Tensor,
    confs: Tensor,
    labels: Tensor,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor]:
    assert xyxys.dim() == 2 and confs.dim() == 1 and labels.dim() == 1, "Input tensors must be 2D, 1D, and 1D"

    labels = labels[confs > conf_threshold]
    xyxys = xyxys[confs > conf_threshold]
    confs = confs[confs > conf_threshold]

    keep = nms(xyxys, confs, iou_threshold=nms_threshold)

    return xyxys[keep], confs[keep], labels[keep]
