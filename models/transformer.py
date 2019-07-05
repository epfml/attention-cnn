import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .bert import BertEncoder, BertConfig
import torchvision.models as models
from torch.autograd import Variable

MAX_WIDTH_HEIGHT = 500


class PositionalEncoding2D(nn.Module):
    """Would be more interesting to use sinusoids instead of learning these embeddings
    """

    def __init__(self, d, max_width_height):
        super().__init__()
        self.d = d
        self.max_width_height = max_width_height
        self.embeddings = Parameter(torch.zeros(max_width_height, max_width_height, d))
        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings.data.normal_(0.0, 1 / self.d)

    def forward(self, X):
        """X should be NWHC format"""
        batch_size, width, height, _ = X.shape
        return self.embeddings[:width, :height].unsqueeze(0)


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
            encodings[:, channel] = torch.ger( torch.sin(positionsX / 10000 ** (channel / x.size(-1))),
                                               torch.sin(positionsY / 10000 ** (channel / x.size(-1))))
        else:
            encodings[:, channel] = torch.ger(torch.cos(positionsX / 10000 ** ((channel - 1) / x.size(-1))),
                                              torch.cos(positionsX / 10000 ** ((channel - 1) / x.size(-1))))
    return Variable(encodings)


class ResBottom(nn.Module):
    def __init__(self,origin_model, block_num=1):
        super(ResBottom, self).__init__()
        self.seq = nn.Sequential(*list(origin_model.children())[0:(4+block_num)])

    def forward(self, batch):
        return self.seq(batch)


class BertImage(nn.Module):
    """
    Wrapper for a Bert encoder
    """

    def __init__(self, config, num_classes, with_resnet=True):
        super().__init__()
        # hard coded
        self.with_resnet = with_resnet
        if with_resnet:
            res50 = models.resnet50(pretrained=True)
            self.extract_feature = ResBottom(res50)
            num_channels_in = 256
        else:
            num_channels_in = 3
        num_channels_out = 3

        self.hidden_size = config["hidden_size"]
        bert_config = BertConfig.from_dict(config)

        self.upscale = nn.Linear(num_channels_in, self.hidden_size)
        #self.positional_encoding = PositionalEncoding2D(self.hidden_size, MAX_WIDTH_HEIGHT)

        self.encoder = BertEncoder(bert_config)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.pixelizer = nn.Linear(self.hidden_size, num_channels_out)
        self.register_buffer("attention_mask", torch.tensor(1.0))

        self.mask_embedding = Parameter(torch.zeros(self.hidden_size))
        self.cls_embedding = Parameter(torch.zeros(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.mask_embedding.data.normal_(mean=0.0, std=0.01)
        self.cls_embedding.data.normal_(mean=0.0, std=0.01)  # TODO no hard coded

    def forward(self, batch_images, batch_mask=None):

        if self.with_resnet:
            orig_resnet = self.extract_feature(batch_images)
            batch_images = self.extract_feature(batch_images)
        batch_size, num_channels_in, width, height = batch_images.shape

        assert (
            width < self.positional_encoding.max_width_height
            and height < self.positional_encoding.max_width_height
        )
        # reshape from NCHW to NHWC
        batch_images = batch_images.permute(0, 2, 3, 1)

        batch_images = self.upscale(batch_images)

        # replace masked pixel with mask "embedding"
        if batch_mask is not None:
            batch_images[~batch_mask] = self.mask_embedding

        # add positional embedding
        #batch_images += self.positional_encoding(batch_images)
        batch_images += positional_encodings_like(batch_images) # 2D sinusoidal position encoding
        """
        # prepend classification token
        data = torch.cat(
            [
                self.cls_embedding.expand(batch_size, 1, -1),
                batch_images.view(batch_size, -1, self.hidden_size),
            ],
            dim=1,
        )
        """

        representations = self.encoder(
            batch_images, attention_mask=self.attention_mask, output_all_encoded_layers=False  # TODO
        )[0]

        cls_representation = representations[:, 0]
        cls_prediction = self.classifier(cls_representation)

        #pix_representation = representations[:, 1:]
        pix_representation = representations
        pix_output = self.pixelizer(pix_representation, batch_mask) # TODO: rewrite pixelizer using the mask
        pix_output = pix_output.reshape(batch_size, width, height, -1)
        # back to NCWH format
        pix_output = pix_output.permute(0, 3, 1, 2)

        return cls_prediction, pix_output, orig_resnet
