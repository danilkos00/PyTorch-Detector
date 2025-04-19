import time
from src.loss import Loss
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torch import nn
from tools.detection_utils import nms


def _display_grad(model):
    total_norm = 0
    max_grad = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())

    total_norm = total_norm ** 0.5
    return total_norm, max_grad


def train_model(model, optimizer, dataloaders, num_epochs):
    """
    Train model for a specified number of epochs with given optimizer and data loaders.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        Optimization algorithm (e.g., Adam, SGD) for model parameter updates.
    dataloaders : dict
        Dictionary containing 'train' and 'val' DataLoader instances:
        - 'train': DataLoader for training data
        - 'val': DataLoader for validation data
    num_epochs : int
        Number of complete passes through the training dataset.
    """
    device = next(model.parameters()).device
    since = time.time()
    best_map = 0.0
    criterion = Loss(neg_ratio=1, loc_weight=1, cls_weight=1, iou_threshold=0.4, alpha=[250.0, 750.0], gamma=2.0)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                metric = MeanAveragePrecision(box_format='cxcywh')
                model.eval()

            running_loss = 0.0
            running_cls_loss = 0.0
            running_loc_loss = 0.0
            total_norm = 0
            max_grad = 0

            pbar = tqdm(dataloaders[phase])

            for images, bboxes, classes in pbar:
                targets = []
                images = images.to(device)
                for i in range(len(bboxes)):
                    bboxes[i] = bboxes[i].to(device)
                    classes[i] = classes[i].to(device)
                    if phase == 'val':
                        targets.append(dict(boxes=bboxes[i],
                                            labels=classes[i]))

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(images)

                    loss, loc_loss, cls_loss = criterion(outputs['pred_offsets'],
                                                        outputs['pred_classes'],
                                                        outputs['default_boxes'], bboxes, classes)


                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                        total_norm, max_grad = _display_grad(model)
                    else:
                        metric.update(nms(outputs['pred_loc'], outputs['pred_classes'], 0.5, 0.5), targets)

                    if len(optimizer.param_groups) > 1:
                        pbar.set_postfix(loss=f'{loss.item():.4f}', loc_loss=f'{loc_loss:.4f}', cls_loss=f'{cls_loss:.4f}', lr_b=optimizer.param_groups[0]["lr"], lr_h=optimizer.param_groups[1]["lr"], gnorm=f'{total_norm:.4e}', gmax=f'{max_grad:.4e}')
                    else:
                        pbar.set_postfix(loss=f'{loss.item():.4f}', loc_loss=f'{loc_loss:.4f}', cls_loss=f'{cls_loss:.4f}', lr=optimizer.param_groups[0]["lr"], gnorm=f'{total_norm:.4e}', gmax=f'{max_grad:.4e}')

                running_cls_loss += cls_loss * images.size(0)
                running_loc_loss += loc_loss * images.size(0)
                running_loss += loss.item() * images.size(0)

            epoch_loc_loss = running_loc_loss / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            epoch_cls_loss = running_cls_loss / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            epoch_loss = epoch_cls_loss + epoch_loc_loss


            if phase == 'val':
                result = metric.compute()
                print(f'val Epoch Loss: {epoch_loss:.4f}, cls Loss: {epoch_cls_loss:.4f}, loc Loss: {epoch_loc_loss:.4f}; mAP: {result["map"].item():.4f}; mAP_50: {result["map_50"].item():.4f}')
            else:
                print(f'train Epoch Loss: {epoch_loss:.4f}, cls Loss: {epoch_cls_loss:.4f}, loc Loss: {epoch_loc_loss:.4f}')

        if result['map_50'].item() > best_map:
            best_map = result['map_50'].item()
            st = model.state_dict()
            torch.save(st, f'FaceDetector_params_{result:.4f}.tar')

        print('-' * 10)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')