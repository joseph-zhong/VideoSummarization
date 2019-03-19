import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.autograd import Function

from torchvision.models import resnet50

import src.utils.utility as _util
import src.train.train_test_utils as _train
from src.data.caption import Token, vocab


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
        return os.path.join(_util.get_weights_path_by_param(reuse=True, dataset="imagenet", model="resnet50"), "weights.pth")

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
        return os.path.join(_util.get_weights_path_by_param(reuse=True, dataset="sports1m", model="c3d"), "weights.pickle")

    @staticmethod
    def feature_size():
        return 4096


# REVIEW josephz: There is some random-ness introduced that is not seeded.
class BinaryGate(Function):
    '''
    二值门单元
    forward中的二值门单元分为train和eval两种：
    train: 阈值为[0,1]内均匀分布的随机采样值随机的二值神经元，
    eval: 固定阈值为0.5的二值神经元
    backward中的二值门单元的导函数用identity函数
    '''

    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        if inplace:
            output = input
        else:
            output = input.clone()
        ctx.thrs = random.uniform(0, 1) if training else 0.5
        output[output > ctx.thrs] = 1
        output[output <= ctx.thrs] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class BoundaryDetector(nn.Module):
    '''
    Boundary Detector，边界检测模块
    '''

    def __init__(self, i_features, h_features, s_features, inplace=False):
        super(BoundaryDetector, self).__init__()
        self.inplace = inplace
        self.Wsi = Parameter(torch.Tensor(s_features, i_features))
        self.Wsh = Parameter(torch.Tensor(s_features, h_features))
        self.bias = Parameter(torch.Tensor(s_features))
        self.vs = Parameter(torch.Tensor(1, s_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Wsi.size(1))
        self.Wsi.data.uniform_(-stdv, stdv)
        self.Wsh.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.vs.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        z = F.linear(x, self.Wsi) + F.linear(h, self.Wsh) + self.bias
        z = F.sigmoid(F.linear(z, self.vs))
        return BinaryGate.apply(z, self.training, self.inplace)

    def __repr__(self):
        return self.__class__.__name__


class Encoder(nn.Module):
    '''
    Hierarchical Boundart-Aware视频编码器
    '''

    def __init__(self, feature_size, projected_size, mid_size, hidden_size, max_frames):
        '''
        feature_size: 视频帧的特征大小，2048维
        projected_size: 特征的投影维度
        mid_size: BD单元的中间表达维度
        hidden_size: LSTM的隐藏单元个数（隐层表示的维度）
        num_frames: 视觉特征的序列长度
        '''
        super(Encoder, self).__init__()

        self.feature_size = feature_size
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.max_frames = max_frames

        # frame_embed用来把视觉特征降维
        self.frame_embed = nn.Linear(feature_size, projected_size)
        self.frame_drop = nn.Dropout(p=0.5)

        # lstm1_cell是低层的视频序列编码单元
        self.lstm1_cell = nn.LSTMCell(projected_size, hidden_size)
        self.lstm1_drop = nn.Dropout(p=0.5)

        # bd是一个边界检测单元
        self.bd = BoundaryDetector(projected_size, hidden_size, mid_size)

        # lstm2_cell是高层的视频序列编码单元
        self.lstm2_cell = nn.LSTMCell(hidden_size, hidden_size, bias=False)
        self.lstm2_drop = nn.Dropout(p=0.5)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return d.data.new(bsz, self.hidden_size).zero_(), d.data.new(bsz, self.hidden_size).zero_()

    def forward(self, video_feats):
        '''
        用Hierarchical Boundary-Aware Neural Encoder对视频进行编码
        '''
        batch_size = len(video_feats)
        # 初始化LSTM状态
        lstm1_h, lstm1_c = self._init_lstm_state(video_feats)
        lstm2_h, lstm2_c = self._init_lstm_state(video_feats)

        # 只取表观特征
        video_feats = video_feats[:, :, :self.feature_size].contiguous()

        v = video_feats.view(-1, self.feature_size)
        v = self.frame_embed(v)
        v = self.frame_drop(v)
        v = v.view(batch_size, -1, self.projected_size)

        for i in range(self.max_frames):
            s = self.bd(v[:, i, :], lstm1_h)
            # print(sum(s.data)[0])
            lstm1_h, lstm1_c = self.lstm1_cell(v[:, i, :], (lstm1_h, lstm1_c))
            lstm1_h = self.lstm1_drop(lstm1_h)

            lstm2_input = lstm1_h * s
            lstm2_h, lstm2_c = self.lstm2_cell(lstm2_input, (lstm2_h, lstm2_c))
            lstm2_h = self.lstm2_drop(lstm2_h)

            lstm1_h = lstm1_h * (1 - s)
            lstm1_c = lstm1_c * (1 - s)

        return lstm2_h


class Decoder(nn.Module):
    '''
    视频内容解码器
    '''

    def __init__(self, encoded_size, projected_size, hidden_size,
                 max_words):
        super(Decoder, self).__init__()
        self.encoded_size = encoded_size
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.max_words = max_words

        self.word_embed = nn.Embedding(len(vocab()), projected_size)
        self.word_drop = nn.Dropout(p=0.5)

        # REVIEW josephz: ??? Understand this.
        # The GRU in the paper has three inputs, except the last hidden layer state of the input GRU.
        # We also need to enter the two dimensions of video features and word features.
        # However, the standard GRU only accepts two inputs
        # Therefore, we use two fully connected layers to merge the two dimensional features
        # into one dimension outside the GRU.
        self.v2m = nn.Linear(encoded_size, projected_size)
        self.w2m = nn.Linear(projected_size, projected_size)
        self.gru_cell = nn.GRUCell(projected_size, hidden_size)
        self.gru_drop = nn.Dropout(p=0.5)
        self.word_restore = nn.Linear(hidden_size, len(vocab()))

    def _init_gru_state(self, d):
        bsz = d.size(0)
        return d.data.new(bsz, self.hidden_size).zero_()

    def forward(self, video_encoded, captions, use_cuda=False, teacher_forcing_ratio=0.5):
        """

        Args:
            video_encoded:
            captions (torch.LongTensor [max_vid_len, max_cap_len]): Caption indices.
            teacher_forcing_ratio:

        Returns:


        """
        batch_size = len(video_encoded)

        # During inference time, caption-labels are not available.
        infer = True if captions is None else False
        if not infer:
            # captions[captions >= len(vocab())] = vocab()[Token.UNK]
            assert captions.max() <= len(vocab())  # REVIEW josephz: Fix this afterwards, with comment on obscure bug

        # Initialize GRU state.
        gru_h = self._init_gru_state(video_encoded)

        if infer:
            if use_cuda:
                outputs = torch.cuda.FloatTensor(self.max_words, batch_size).fill_(0)
            else:
                outputs = torch.FloatTensor(self.max_words, batch_size).fill_(0)
            outputs[0] = vocab()[Token.START]
        else:
            if use_cuda:
                outputs = torch.cuda.FloatTensor(self.max_words, batch_size, len(vocab())).fill_(0)
            else:
                outputs = torch.FloatTensor(self.max_words, batch_size, len(vocab())).fill_(0)


        # Append START token to to sentence.
        word_id = vocab()[Token.START]
        word = video_encoded.data.new(batch_size, 1).long().fill_(word_id)
        word = self.word_embed(word).squeeze(1)
        word = self.word_drop(word)

        vm = self.v2m(video_encoded)
        for i in range(self.max_words):
            if not infer:
                allThings = True
                for x in captions[:, i]:
                    if x != vocab()[Token.PAD]:
                        allThings = False
                if allThings:
                    break

            # if not infer and all(x == v[Token.PAD] for x in captions[:, i].data):
            #     If all the word ids are Token.PAD, then we have hit the end of the sentence.
                # break
            # Push word to decoder.
            wm = self.w2m(word)

            # Concatenate the video encoding and word encoding.
            m = vm + wm
            gru_h = self.gru_cell(m, gru_h)
            gru_h = self.gru_drop(gru_h)

            word_logits = self.word_restore(gru_h)
            use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)
            if use_teacher_forcing:
                word_id = captions[:, i].clone()
            else:
                word_id = word_logits.max(1)[1]

            if infer:
                # In infer mode, use word_id from label.
                outputs[i] = word_id
            else:
                # Otherwise, generate word from logits.
                outputs[i] = word_logits
            # Compute word representation.
            word = self.word_embed(word_id).squeeze(1)
            word = self.word_drop(word)
        # unsqueeze(1)会把一个向量(n)拉成列向量(nx1)
        # outputs中的每一个向量都是整个batch在某个时间步的输出
        # 把它拉成列向量之后再横着拼起来，就能得到整个batch在所有时间步的输出
        assert len(outputs) > 0
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1).contiguous()
        return outputs

    def sample(self, video_feats):
        '''
        sample就是不给caption且不用teacher forcing的forward
        '''
        return self.forward(video_feats, None, teacher_forcing_ratio=0.0)

class BANet(nn.Module):
    def __init__(self, feature_size, projected_size, mid_size, hidden_size,
                 max_frames, max_words, use_cuda):
        super(BANet, self).__init__()

        encoder = Encoder(feature_size, projected_size, mid_size, hidden_size, max_frames)
        decoder = Decoder(hidden_size, projected_size, hidden_size, max_words)
        if use_cuda:
            encoder = _train.DataParallel(encoder)
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda

    def forward(self, videos, captions, use_cuda=False, teacher_forcing_ratio=0.5):
        video_encoded = self.encoder(videos)
        output = self.decoder(video_encoded, captions, self.use_cuda, teacher_forcing_ratio)
        return output, video_encoded
