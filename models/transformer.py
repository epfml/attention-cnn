import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from .bert import BertEncoder, BertConfig
import torchvision.models as models
from torch.autograd import Variable
from enum import Enum
from .positional_encoding import PositionalEncodingType, PositionalEncoding

import timer


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
        self.timer = timer.default()

        self.with_resnet = config["use_resnet"]
        self.hidden_size = config["hidden_size"]
        self.concat_pooling = config["concat_pooling"]
        assert (config["concat_pooling"] == 1) or (
            not config["use_resnet"]
        ), "Use either resnet or concat_pooling"

        self.positional_encoding = PositionalEncoding(
            config["positional_encoding"], self.hidden_size
        )

        if self.with_resnet:
            res50 = models.resnet50(pretrained=True)
            self.extract_feature = ResBottom(res50)

            # compute downscale factor and channel at output of ResNet
            _, num_channels_in, new_width, new_height = self.extract_feature(
                torch.rand(1, 3, 1024, 1024)
            ).shape
            self.feature_downscale_factor = 1024 // new_width
        elif self.concat_pooling > 1:
            num_channels_in = 3 * (self.concat_pooling ** 2)
        else:
            num_channels_in = 3

        bert_config = BertConfig.from_dict(config)

        self.features_upscale = nn.Linear(num_channels_in, self.hidden_size)
        self.features_downscale = nn.Linear(self.hidden_size, num_channels_in)

        self.encoder = BertEncoder(bert_config)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.pixelizer = nn.Linear(self.hidden_size, 3)
        self.register_buffer("attention_mask", torch.tensor(1.0))

        self.mask_embedding = Parameter(torch.zeros(self.hidden_size))
        self.cls_embedding = Parameter(torch.zeros(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.mask_embedding.data.normal_(mean=0.0, std=0.01)
        self.cls_embedding.data.normal_(mean=0.0, std=0.01)  # TODO no hard coded
        self.positional_encoding.reset_parameters()

    def random_masking(self, batch_images, batch_mask, device):
        """
        with probability 10% we keep the image unchanged;
        with probability 10% we change the mask region to a normal distribution
        with 80% we mask the region as 0.
        :param batch_images: image to be masked
        :param batch_mask: mask region
        :param device:
        :return: masked image
        """
        return batch_images
        # TODO disabled
        temp = random.random()
        if temp > 0.1:
            batch_images = batch_images * batch_mask.unsqueeze(1).float()
            if temp < 0.2:
                batch_images = batch_images + (
                    ((-batch_mask.unsqueeze(1).float()) + 1)
                    * torch.normal(mean=0.5, std=torch.ones(batch_images.shape)).to(device)
                )
        return batch_images

    def forward(self, batch_images, batch_mask=None, feature_mask=None, device=None):

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
                    batch_images = self.random_masking(batch_images, batch_mask, device)
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
            # reshape from NCHW to NHWC
            batch_features = batch_features.permute(0, 2, 3, 1)

        elif self.concat_pooling > 1:

            def downsample_concatenate(X, kernel):
                """X is of shape B x H x W x C
                return shape B x (kernel*H) x (kernel*W) x (kernel*kernel*C)
                """
                b, h, w, c = X.shape
                Y = X.contiguous().view(b, h, w // kernel, c * kernel)
                Y = Y.permute(0, 2, 1, 3).contiguous()
                Y = Y.view(b, w // kernel, h // kernel, kernel * kernel * c).contiguous()
                Y = Y.permute(0, 2, 1, 3).contiguous()
                return Y

            # reshape from NCHW to NHWC
            batch_features = batch_images.permute(0, 2, 3, 1)
            batch_features = downsample_concatenate(batch_features, self.concat_pooling)
            feature_mask = None
            if batch_mask is not None:
                feature_mask = batch_mask[:, :: self.concat_pooling, :: self.concat_pooling]

        else:
            batch_features = batch_images
            feature_mask = batch_mask
            # reshape from NCHW to NHWC
            batch_features = batch_features.permute(0, 2, 3, 1)

        # feature upscale to BERT dimension
        with self.timer("upscale"):
            batch_features = self.features_upscale(batch_features)

        # replace masked "pixels" by [MSK] token
        if feature_mask is not None:
            batch_features[~feature_mask] = self.mask_embedding

        # add positional embedding
        batch_features = self.positional_encoding(batch_features)

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
