import torchvision.ops as ops
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from albumentations import Resize
import matplotlib.pyplot as plt


def nms(pred_boxes, confidences, iou_threshold=0.5, score_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes in format [cx,cy,w,h].
        Shape: [B, N, 4] where B is the batch size, N is the number of boxes.
    confidences : torch.Tensor
        Confidence scores for each predicted box (typically class probabilities).
        Shape: [B, N, C] where B is the batch size, N is number of boxes and C is number of classes.
    iou_threshold : float, optional
        Intersection-over-Union threshold for box suppression (default: 0.5).
        Boxes with IoU > threshold with a higher-scoring box will be suppressed.
    score_threshold : float, optional
        Minimum confidence score to consider a box (default: 0.5).
        Boxes with confidence < threshold are discarded before NMS.

    Returns
    -------
    list[dict]
        A list where each element is a dictionary containing filtered predictions
        for a class, with keys:
        - 'boxes': torch.Tensor of shape [M, 4] (filtered boxes in [cx,cy,w,h] format)
        - 'scores': torch.Tensor of shape [M] (confidence scores for kept boxes)
        - 'labels': torch.Tensor of shape [M] (class labels as long integers)
        where M is the number of boxes remaining after NMS for that class.
    """
    device = pred_boxes.device
    confidences = F.softmax(confidences, dim=-1)

    batch_size = pred_boxes.size(0)
    num_classes = confidences.size(2)

    all_filtered = []

    for i in range(batch_size):
        boxes = box_convert(pred_boxes[i], 'cxcywh', 'xyxy')
        scores = confidences[i]

        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []

        for class_idx in range(1, num_classes):
            class_scores = scores[:, class_idx]
            mask = class_scores > score_threshold

            if mask.sum() == 0:
                continue

            class_boxes = boxes[mask]
            class_scores = class_scores[mask]

            keep = ops.nms(class_boxes, class_scores, iou_threshold)

            filtered_boxes.append(box_convert(class_boxes[keep], 'xyxy', 'cxcywh'))
            filtered_labels.append(torch.full_like(class_scores[keep], class_idx))
            filtered_scores.append(class_scores[keep])

        if len(filtered_boxes) > 0:
            all_filtered.append(dict(boxes=torch.cat(filtered_boxes, dim=0),
                               scores=torch.cat(filtered_scores, dim=0),
                               labels=torch.cat(filtered_labels, dim=0).long()))
        else:
            all_filtered.append(dict(boxes=torch.zeros((0, 4), device=device),
                               scores=torch.zeros(0, dtype=torch.long, device=device),
                               labels=torch.zeros(0, device=device)))

    return all_filtered


def resize_back(bboxes, original_size, resize_size=300):
    original_h, original_w = original_size
    scale = resize_size / max(original_h, original_w)
    resized_h = int(original_h * scale)
    resized_w = int(original_w * scale)

    pad_y = (resize_size - resized_h) // 2
    pad_x = (resize_size - resized_w) // 2


    bboxes = box_convert(bboxes, 'cxcywh', 'xyxy') * resize_size

    recovered_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox.tolist()

        xmin -= pad_x
        xmax -= pad_x
        ymin -= pad_y
        ymax -= pad_y

        xmin /= scale
        xmax /= scale
        ymin /= scale
        ymax /= scale

        xmin = max(0, min(xmin, original_w))
        xmax = max(0, min(xmax, original_w))
        ymin = max(0, min(ymin, original_h))
        ymax = max(0, min(ymax, original_h))

        xmin /= original_w
        xmax /= original_w
        ymin /= original_h
        ymax /= original_h

        recovered_bboxes.append([xmin, ymin, xmax, ymax])

    recovered_bboxes = box_convert(torch.tensor(recovered_bboxes), 'xyxy', 'cxcywh')

    return recovered_bboxes


def imshow(image, bboxes=None, color='green', figsize=(5, 5), box_width=1):
    """
    Visualize image with matplotlib.

    Parameters
    ----------
    image : torch.Tensor
        image to display
    bboxes : torch.Tensor, optional
        Bounding boxes in for image in [cx, cy, w, h] format (default: None).
    color : str, optional
        Color of bounding boxes (defaul: 'green').
    figsize : tuple[float, float], optional
        Width, height for matplotlib figure (default: (5, 5)).
    box_width : float, optianal
        Width of bounding boxes (default: 1).
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = std * image.permute(1, 2, 0) + mean
    plt.figure(figsize=figsize)
    if bboxes is not None:
        bboxes = box_convert(bboxes, 'cxcywh', 'xyxy')
        bboxes[..., [0, 2]] *= image.size(1)
        bboxes[..., [1, 3]] *= image.size(0)
        image = draw_bounding_boxes(image.permute(2, 0, 1), bboxes, colors=color, width=box_width)
        plt.imshow(image.permute(1, 2, 0))
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.pause(0.01)


def show_preds(model, dataset, num_images=1, color='blue', box_width=1):
    """
    Visualize model predictions on random images from a dataset with matplotlib.

    Parameters
    ----------
    model : torch.nn.Module
        Trained detection model that implements a forward pass returning predictions.
    dataset : torch.utils.data.Dataset
        Dataset object containing images and annotations.
    num_images : int, optional
        Number of random images to display (default: 1)
    color : str, optional
        Color of bounding boxes (default: 'blue')
    box_width : float, optional
        Width of bounding boxes (default: 1)
    """
    model.eval()
    device = next(model.parameters()).device
    m = len(dataset)
    for _ in range(num_images):
        i = torch.randint(m, (1,)).item()
        with torch.no_grad():
            images, bboxes, classes = dataset[i]
            bboxes = bboxes.to(device)
            classes = classes.to(device)
            outputs = model(images.unsqueeze(0).to(device))
            imshow(images, nms(outputs['pred_loc'], outputs['pred_classes'], 0.2, 0.5)[0]['boxes'], color=color, box_width=box_width)
