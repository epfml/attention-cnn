import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from .bert import BertEncoder, BertConfig
import torchvision.models as models
from torch.autograd import Variable
from enum import Enum

import timer

MAX_WIDTH_HEIGHT = 500


class PositionalEncodingType(Enum):
    Sinusoid2d = "Sinusoid2d"
    Learned = "Learned"
    Relative = "Relative"


def positional_encodings_like(x, t=None):
    if t is None:
        positionsX = torch.arange(0, x.size(1)).float()
        positionsY = torch.arange(0, x.size(2)).float()
        if x.is_cuda:
            positionsX = positionsX.cuda(x.get_device())
            positionsY = positionsY.cuda(x.get_device())
    else:
        positionsX, positionsY = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.ger(
                torch.sin(positionsX / 10000 ** (channel / x.size(-1))),
                torch.sin(positionsY / 10000 ** (channel / x.size(-1))),
            )
        else:
            encodings[:, channel] = torch.ger(
                torch.cos(positionsX / 10000 ** ((channel - 1) / x.size(-1))),
                torch.cos(positionsX / 10000 ** ((channel - 1) / x.size(-1))),
            )
    return Variable(encodings)


def positional_encodings_concat(x, t=None):
    if t is None:
        positionsX = torch.arange(0, x.size(1)).float().unsqueeze(-1).expand(-1, x.size(2))
        positionsY = torch.arange(0, x.size(2)).float().unsqueeze(0).expand(x.size(1), -1)
        if x.is_cuda:
            positionsX = positionsX.cuda(x.get_device())
            positionsY = positionsY.cuda(x.get_device())
    else:
        positionsX, positionsY = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    midchannel = int(x.size(-1) / 2)
    for channel in range(midchannel):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(positionsX / 10000 ** (channel / midchannel))
            encodings[:, channel + midchannel] = torch.sin(
                positionsY / 10000 ** (channel / midchannel)
            )
        else:
            encodings[:, channel] = torch.sin(positionsX / 10000 ** (channel / midchannel))
            encodings[:, channel + midchannel] = torch.sin(
                positionsY / 10000 ** (channel / midchannel)
            )
    return Variable(encodings)


class ResBottom(nn.Module):
    def __init__(self, origin_model, block_num=1):
        super(ResBottom, self).__init__()
        self.seq = nn.Sequential(*list(origin_model.children())[0 : (4 + block_num)])

    def forward(self, batch):
        return self.seq(batch)


class BertImage(nn.Module):
    """
    Wrapper for a Bert encoder
    """

    def __init__(self, config, num_classes):
        super().__init__()
        self.with_resnet = config["use_resnet"]
        self.positional_encoding_type = config["positional_encoding"]
        self.timer = timer.default()

        if self.with_resnet:
            res50 = models.resnet50(pretrained=True)
            self.extract_feature = ResBottom(res50)

            # compute downscale factor and channel at output of ResNet
            _, num_channels_in, new_width, new_height = self.extract_feature(
                torch.rand(1, 3, 1024, 1024)
            ).shape
            self.feature_downscale_factor = 1024 // new_width
        else:
            num_channels_in = 3

        self.hidden_size = config["hidden_size"]
        bert_config = BertConfig.from_dict(config)

        self.features_upscale = nn.Linear(num_channels_in, self.hidden_size)
        self.features_downscale = nn.Linear(self.hidden_size, num_channels_in)

        self.encoder = BertEncoder(bert_config)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.pixelizer = nn.Linear(self.hidden_size, 3)
        self.register_buffer("attention_mask", torch.tensor(1.0))

        # positional encoding
        if self.positional_encoding_type == PositionalEncodingType.Learned:
            self.positional_encoding = Parameter(
                torch.zeros(MAX_WIDTH_HEIGHT, MAX_WIDTH_HEIGHT, self.hidden_size)
            )
            # will be initialized randomly in reset_parameters
        elif self.positional_encoding_type == PositionalEncodingType.Sinusoid2d:
            # 2D positional endoding as 2D sinusoid of different frequencies
            # coded as Vaswani et al. Attention is all you need
            positional_encoding = torch.zeros(MAX_WIDTH_HEIGHT, MAX_WIDTH_HEIGHT, self.hidden_size)
            assert (
                self.hidden_size % 2 == 0
            ), "hidden size should be even for half cos/sin positional encodings"
            d = self.hidden_size // 2

            r = torch.sqrt(
                (
                    torch.arange(MAX_WIDTH_HEIGHT).view(-1, 1) ** 2
                    + torch.arange(MAX_WIDTH_HEIGHT).view(1, -1) ** 2
                ).float()
            )
            wavelenghts = 10000 ** (torch.arange(d).float() / (d - 1)).view(1, 1, -1)
            positional_encoding[:, :, :d] = torch.sin(r.unsqueeze(-1) / wavelenghts)
            positional_encoding[:, :, d:] = torch.cos(r.unsqueeze(-1) / wavelenghts)
            self.register_buffer("positional_encoding", positional_encoding)

        self.mask_embedding = Parameter(torch.zeros(self.hidden_size))
        self.cls_embedding = Parameter(torch.zeros(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.mask_embedding.data.normal_(mean=0.0, std=0.01)
        self.cls_embedding.data.normal_(mean=0.0, std=0.01)  # TODO no hard coded

        if self.positional_encoding_type == PositionalEncodingType.Learned:
            self.positional_encoding.data.normal_(0.0, 1 / self.hidden_size)

    def positional_encodings_like(self, X):
        batch_size, width, height, channels = X.shape  # check channel order
        return self.positional_encoding[:width, :height, :].unsqueeze(0)  # unsqueeze the batch size

    def forward(self, batch_images, batch_mask=None, feature_mask=None):

        """
        Replace masked pixels with 0s
        If ResNet
        | compute features
        | downscale the mask
        Replace masked pixels/features by MSK token
        Use Bert encoder
        """
        # compute ResNet features
        if self.with_resnet:

            with self.timer("resnet"):
                # replace masked pixels with 0, batch_images has NCHW format
                batch_features_unmasked = self.extract_feature(batch_images)

                if batch_mask is not None:
                    temp = random.random()
                    if temp > 0.1:
                        batch_images = batch_images * batch_mask.unsqueeze(1).float()
                        if temp < 0.2:
                            batch_images = batch_images + (((-batch_mask.unsqueeze(1).float())+1)*torch.normal(mean=0.5,
                                                                                    std=torch.ones(batch_images.shape)))
                    batch_features = self.extract_feature(batch_images)
                else:
                    batch_features = batch_features_unmasked

                # downscale the mask
                if batch_mask is not None:
                    # downsample the mask
                    # mask any downsampled pixel if it contained one masked pixel originialy
                    feature_mask = ~(
                        F.max_pool2d((~batch_mask).float(), self.feature_downscale_factor).byte()
                    )
        else:
            batch_features = batch_images
            feature_mask = batch_mask

        # reshape from NCHW to NHWC

        with self.timer("permute NCHW to NHWC"):
            batch_features = batch_features.permute(0, 2, 3, 1)

        # feature upscale to BERT dimension
        with self.timer("upscale"):
            batch_features = self.features_upscale(batch_features)

        # replace masked "pixels" by [MSK] token
        if feature_mask is not None:
            batch_features[~feature_mask] = self.mask_embedding

        # add positional embedding
        batch_size, num_channels_in, width, height = batch_features.shape
        assert width < MAX_WIDTH_HEIGHT and height < MAX_WIDTH_HEIGHT
        if not (self.positional_encoding_type==PositionalEncodingType.Relative):
            batch_features += self.positional_encodings_like(batch_features)

        # replace classification token (top left pixel)
        batch_features[:, 0, 0, :] = self.cls_embedding.view(1, -1)

        with self.timer("Bert encoder"):
            representations = self.encoder(
                batch_features,
                attention_mask=self.attention_mask,
                output_all_encoded_layers=False,  # TODO
            )[0]

        cls_representation = representations[:, 0, 0, :]
        cls_prediction = self.classifier(cls_representation)

        with self.timer("downscale"):
            representations = self.features_downscale(representations)

        with self.timer("permute to NCWH"):
            # back to NCWH format
            representations = representations.permute(0, 3, 1, 2)

        if self.with_resnet:
            return cls_prediction, representations, batch_features_unmasked, feature_mask
        else:
            return cls_prediction, representations
