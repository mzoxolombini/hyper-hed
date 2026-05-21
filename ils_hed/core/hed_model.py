"""
HED Network model wrapper for ILS-HED.
Contains: HEDNetwork class (VGG-based HED with 5 side outputs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class HEDNetwork(nn.Module):
    """Official HED architecture with 5 side outputs (S1-S5) - LOCKED"""

    def __init__(self, pretrained_path: Optional[str] = None):
        super(HEDNetwork, self).__init__()

        # VGG-16 stages
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Side outputs
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)

        self.fuse_weight = nn.Parameter(torch.ones(5) / 5)
        self._initialize_weights()

        if pretrained_path and __import__('os').path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5]:
                    nn.init.normal_(m.weight, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded HED weights: {len(pretrained_dict)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"Warning: Could not load HED weights: {e}")

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        d1 = F.interpolate(self.dsn1(conv1), size=(h, w), mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dsn2(conv2), size=(h, w), mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.dsn3(conv3), size=(h, w), mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.dsn4(conv4), size=(h, w), mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.dsn5(conv5), size=(h, w), mode='bilinear', align_corners=False)

        d1, d2, d3, d4, d5 = map(torch.sigmoid, [d1, d2, d3, d4, d5])

        fuse_weights = F.softmax(self.fuse_weight, dim=0)
        fuse = d1 * fuse_weights[0] + d2 * fuse_weights[1] + d3 * fuse_weights[2] + \
               d4 * fuse_weights[3] + d5 * fuse_weights[4]
        fuse = torch.sigmoid(fuse)

        return [d1, d2, d3, d4, d5, fuse]

    @torch.no_grad()
    def extract_side_outputs(self, image_np: np.ndarray) -> List[np.ndarray]:
        """Extract the 5 side outputs (S1-S5) - LOCKED features"""
        self.eval()
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        if image_np.max() > 1:
            image_np = image_np / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(next(self.parameters()).device)

        outputs = self.forward(image_tensor)
        return [o.squeeze().cpu().numpy() for o in outputs[:5]]
