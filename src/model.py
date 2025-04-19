import torch
from torch import nn
import torch.nn.functional as F
from itertools import product
from torchvision import models
from math import sqrt


class L2NormLayer(nn.Module):
    def __init__(self, n_channels, scale=20.0, eps=1e-10):
        super(L2NormLayer, self).__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.full((n_channels,), scale))

    def forward(self, x):
        x_normalized = F.normalize(x, p=2, dim=1, eps=self.eps)
        x_scaled = x_normalized * self.weight.view(1, self.n_channels, 1, 1)
        return x_scaled


class DetectorSSD(nn.Module):
    def __init__(self, num_classes):
        super(DetectorSSD, self).__init__()

        backbone = models.resnext50_32x4d(weights=None)

        self.num_classes = num_classes + 1

        self.scaler = L2NormLayer(n_channels=512)

        self.feature_extractor = nn.ModuleList()


        self.feature_extractor.add_module('feature_extractor1',
                                          nn.Sequential(*list(backbone.children())[:-4],
                                                       nn.Conv2d(512, 256, (3, 3), padding=1),
                                                       nn.ReLU(inplace=True),
                                                       nn.Conv2d(256, 512, (3, 3), padding=2, dilation=2),
                                                       nn.ReLU(inplace=True)))

        self.feature_extractor.add_module('feature_extractor2',
                                          nn.Sequential(
                                              *list(backbone.children())[-4:-2],
                                              nn.Conv2d(2048, 1024, (3, 3), padding=6),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(1024, 1024, (1, 1)),
                                              nn.ReLU(inplace=True)))

        self.feature_extractor.add_module('extra_layer1',
                                          nn.Sequential(
                                              nn.Conv2d(1024, 256, (3, 3), padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 512, (1, 1), stride=2),
                                              nn.ReLU(inplace=True)))

        self.feature_extractor.add_module('extra_layer2',
                                          nn.Sequential(
                                              nn.Conv2d(512, 128, (3, 3), padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 256, (1, 1), stride=2),
                                              nn.ReLU(inplace=True)))

        self.feature_extractor.add_module('extra_layer3',
                                          nn.Sequential(
                                              nn.Conv2d(256, 128, (3, 3)),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 256, (1, 1)),
                                              nn.ReLU(inplace=True)))

        self.feature_extractor.add_module('extra_layer4',
                                           nn.Sequential(
                                              nn.Conv2d(256, 128, (3, 3)),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 256, (1, 1)),
                                              nn.ReLU(inplace=True)))

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(512, 256, (1, 1)),
            nn.Conv2d(1024, 256, (1, 1)),
            nn.Conv2d(512, 256, (1, 1))
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        ])


        self.regression_head = nn.ModuleList([
            nn.Conv2d(512, 16, (3, 3), padding=1),
            nn.Conv2d(1024, 24, (3, 3), padding=1),
            nn.Conv2d(512, 24, (3, 3), padding=1),
            nn.Conv2d(256, 16, (3, 3), padding=1),
            nn.Conv2d(256, 16, (3, 3), padding=1),
            nn.Conv2d(256, 16, (3, 3), padding=1)
        ])

        self.classification_head = nn.ModuleList([
            nn.Conv2d(512, 4 * self.num_classes, (3, 3), padding=1),
            nn.Conv2d(1024, 6 * self.num_classes, (3, 3), padding=1),
            nn.Conv2d(512, 6 * self.num_classes, (3, 3), padding=1),
            nn.Conv2d(256, 4 * self.num_classes, (3, 3), padding=1),
            nn.Conv2d(256, 4 * self.num_classes, (3, 3), padding=1),
            nn.Conv2d(256, 4 * self.num_classes, (3, 3), padding=1)
        ])

        self.feat_sizes = [38, 20, 10, 5, 3, 1]
        self.aspect_ratious = [[1, 0.5, 0.6],
                               [1, 0.85, 0.7, 0.5, 0.6],
                               [1, 0.85, 0.7, 0.5, 0.6],
                               [1, 0.5, 0.7],
                               [1, 0.5, 0.7],
                               [1, 0.5, 0.7, 0.6]]
        self.prior_scales = [0.04, 0.142, 0.244, 0.346, 0.448, 0.55]

        self.default_boxes = self._generate_default_boxes()


    def _generate_default_boxes(self):
        default_boxes = []
        for idx, feat_size in enumerate(self.feat_sizes):
            for i, j in product(range(feat_size), repeat=2):
                cx = (i + 0.5) / feat_size
                cy = (j + 0.5) / feat_size

                for ar in self.aspect_ratious[idx]:
                    w = self.prior_scales[idx] * sqrt(ar)
                    h = self.prior_scales[idx] / sqrt(ar)

                    default_boxes.append([cx, cy, w, h])

                if idx != 5:
                    extra_wh = sqrt(self.prior_scales[idx] * self.prior_scales[idx + 1])
                    default_boxes.append([cx, cy, extra_wh, extra_wh])

        return torch.tensor(default_boxes)


    def _decode_bboxes(self, default_boxes, predicted_offsets):
        variances = [0.1, 0.2]

        prediction = torch.empty_like(default_boxes)
        prediction[..., 0] = default_boxes[..., 0] + variances[0] * default_boxes[..., 2] * predicted_offsets[..., 0]
        prediction[..., 1] = default_boxes[..., 1] + variances[0] * default_boxes[..., 3] * predicted_offsets[..., 1]
        prediction[..., 2] = default_boxes[..., 2] * torch.exp(variances[1] * predicted_offsets[..., 2])
        prediction[..., 3] = default_boxes[..., 3] * torch.exp(variances[1] * predicted_offsets[..., 3])

        return prediction


    def forward(self, x):
        feature_maps = []
        x = list(self.feature_extractor)[0](x)
        feature_maps.append(self.scaler(x))
        for layer in list(self.feature_extractor)[1:]:
            x = layer(x)
            feature_maps.append(x)

        for i, lat_conv in enumerate(self.lateral_convs):
            feature_maps[i] = lat_conv(feature_maps[i])

        for i in range(2, -1, -1):
            feature_maps[i] = feature_maps[i] + nn.Upsample(feature_maps[i].size()[2:],
                                                            mode='bilinear', align_corners=False)(feature_maps[i + 1])

        for i, smooth_conv in enumerate(self.smooth_convs):
            feature_maps[i] = smooth_conv(feature_maps[i])

        offsets = []
        predicted_classes = []
        for i, (offset_conv, classes_conv) in enumerate(zip(self.regression_head, self.classification_head)):
            offset = offset_conv(feature_maps[i])
            b, _, w, h = offset.size()
            offset = offset.view(b, -1, 4, w, h)
            offset = offset.permute(0, 3, 4, 1, 2)
            offset = offset.reshape(b, -1, 4)
            offsets.append(offset)

            classes = classes_conv(feature_maps[i])
            b, _, w, h = classes.size()
            classes = classes.view(b, -1, self.num_classes, w, h)
            classes = classes.permute(0, 3, 4, 1, 2)
            classes = classes.reshape(b, -1, self.num_classes)
            predicted_classes.append(classes)

        offsets = torch.cat(offsets, dim=1)
        predicted_classes = torch.cat(predicted_classes, dim=1)

        default_boxes = self.default_boxes.to(offset.device)
        default_boxes = torch.stack([default_boxes for _ in range(offsets.size(0))])

        return {'pred_offsets' : offsets,
                'pred_classes' : predicted_classes,
                'pred_loc' : self._decode_bboxes(default_boxes, offsets),
                'default_boxes' : default_boxes}