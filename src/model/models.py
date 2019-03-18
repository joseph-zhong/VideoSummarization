import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import src.utils.utility as _util


class AppearanceEncoder(nn.Module):
    def __init__(self):
        super(AppearanceEncoder, self).__init__()
        self.resnet = resnet50()
        self.resnet.load_state_dict(torch.load(self._get_weights_path()))
        del self.resnet.fc

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    @staticmethod
    def _get_weights_path():
        return os.path.join(_util.getWeightsByParams(reuse=True, dataset="imagenet", model="resnet50"), "weights.pth")

    @staticmethod
    def feature_size():
        return 2048

class C3D(nn.Module):
    """
    C3D model (https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py)
    """

    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        return x


class MotionEncoder(nn.Module):

    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.c3d = C3D()
        pretrained_dict = torch.load(self._get_weights_path())
        model_dict = self.c3d.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.c3d.load_state_dict(model_dict)

    def forward(self, x):
        return self.c3d.forward(x)

    @staticmethod
    def _get_weights_path():
        return os.path.join(_util.getWeightsByParams(reuse=True, dataset="sports1m", model="c3d"), "weights.pickle")

    @staticmethod
    def feature_size():
        return 4096

class BANet(nn.Module):
    def __init__(self, feature_size, projected_size, mid_size, hidden_size,
                 max_frames, max_words, vocab):
        super(BANet, self).__init__()
        self.encoder = Encoder(feature_size, projected_size, mid_size, hidden_size,
                               max_frames)
        self.decoder = Decoder(hidden_size, projected_size, hidden_size,
                               max_words, vocab)

    def forward(self, videos, captions, teacher_forcing_ratio=0.5):
        video_encoded = self.encoder(videos)
        output = self.decoder(video_encoded, captions, teacher_forcing_ratio)
        return output, video_encoded
