import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou


class Loss(nn.Module):
    def __init__(self, neg_ratio=3, loc_weight=1.0, cls_weight=1.0, iou_threshold=0.5):
        super(Loss, self).__init__()

        self.neg_ratio = neg_ratio

        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        
        self.iou_threshold = iou_threshold


    def _encode_bboxes(self, gt_boxes, default_boxes):
        variances = [0.1, 0.2]

        offsets = torch.empty_like(gt_boxes)
        offsets[:, 0] = (gt_boxes[:, 0] - default_boxes[:, 0]) / (variances[0] * default_boxes[:, 2])
        offsets[:, 1] = (gt_boxes[:, 1] - default_boxes[:, 1]) / (variances[0] * default_boxes[:, 3])
        offsets[:, 2] = torch.log(gt_boxes[:, 2] / default_boxes[:, 2]) / variances[1]
        offsets[:, 3] = torch.log(gt_boxes[:, 3] / default_boxes[:, 3]) / variances[1]

        return offsets


    def _matching(self, default_boxes, offsets, gt_loc, pred_classes, gt_classes):
        batch_size = len(gt_loc)
        pos_mask = []

        matched_gt_cls = []
        matched_gt_loc = []
        matched_neg_cls = []

        for batch in range(batch_size):
            iou = box_iou(box_convert(default_boxes[batch], 'cxcywh', 'xyxy'),
                          box_convert(gt_loc[batch], 'cxcywh', 'xyxy'))

            max_iou, max_iou_idx = iou.max(dim=1)
            mask = max_iou > self.iou_threshold

            if not mask.any():
                for gt in gt_loc[batch]:
                    distances = torch.cdist(default_boxes[batch], gt.unsqueeze(0))
                    min_idx = distances.argmin()

                    mask[min_idx] = True

            neg_pred = self._hard_neg_mining(pred_classes[batch], ~mask)

            pos_mask.append(mask)

            matched_neg_cls.append(neg_pred)
            matched_gt_cls.append(gt_classes[batch][max_iou_idx[mask]])

            matched_gt_loc.append(gt_loc[batch][max_iou_idx[mask]])

        pos_mask = torch.stack(pos_mask)

        matched_neg_cls = torch.cat(matched_neg_cls)
        matched_gt_loc = torch.cat(matched_gt_loc)
        matched_gt_cls = torch.cat(matched_gt_cls)
        matched_offsets = offsets[pos_mask]
        matched_priors = default_boxes[pos_mask]
        matched_pred_cls = pred_classes[pos_mask]

        return {'priors' : matched_priors,
                'offsets' : matched_offsets,
                'gt_loc' : matched_gt_loc,
                'pos_cls' : matched_pred_cls,
                'neg_cls' : matched_neg_cls,
                'gt_cls' : matched_gt_cls}, pos_mask


    def _hard_neg_mining(self, pred_classes, neg_mask):
        negative_preds = pred_classes[neg_mask]
        target = torch.zeros(negative_preds.size(0), dtype=torch.long, device=negative_preds.device)
        loss = F.cross_entropy(negative_preds, target, reduction='none')

        loss, idx = loss.sort(descending=True)
        n_positives = (~neg_mask).sum().item()
        n_hard = int(self.neg_ratio * n_positives)

        return negative_preds[idx[:n_hard]]


    def forward(self, pred_offsets, pred_classes, default_boxes, gt_loc, gt_classes):
        matched, pos_mask = self._matching(default_boxes, pred_offsets, gt_loc, pred_classes, gt_classes)

        gt_offsets = self._encode_bboxes(matched['gt_loc'], matched['priors'])

        cls_loss = F.cross_entropy(matched['pos_cls'], matched['gt_cls'], reduction='sum')
        cls_loss = cls_loss + F.cross_entropy(matched['neg_cls'], torch.zeros(matched['neg_cls'].size(0),
                                                                  dtype=torch.long, device=matched['neg_cls'].device), reduction='sum')

        cls_loss = cls_loss / pos_mask.sum().float()

        loc_loss = F.smooth_l1_loss(matched['offsets'], gt_offsets, reduction='sum') / pos_mask.sum().float()

        return self.loc_weight * loc_loss + self.cls_weight * cls_loss, loc_loss.item(), cls_loss.item()