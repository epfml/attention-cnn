from enum import Enum
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

MAX_WIDTH_HEIGHT = 500


class PositionalEncodingType(Enum):
    Sinusoid2d = "Sinusoid2d"
    Learned = "Learned"
    Relative = "Relative"
    Nothing = "Nothing"

    def is_added_to_input(self):
        return self in [PositionalEncodingType.Sinusoid2d, PositionalEncodingType.Learned]

    def is_relative(self):
        return self in [PositionalEncodingType.Relative]


class PositionalEncoding(nn.Module):
    def __init__(self, encoding_type: PositionalEncodingType, hidden_size: int):
        super().__init__()

        self.encoding_type = encoding_type
        self.hidden_size = hidden_size

        if self.encoding_type == PositionalEncodingType.Learned:
            self.positional_encoding = Parameter(
                torch.zeros(MAX_WIDTH_HEIGHT, MAX_WIDTH_HEIGHT, self.hidden_size)
            )
            # will be initialized randomly in reset_parameters

        elif self.encoding_type == PositionalEncodingType.Sinusoid2d:
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

    def reset_parameters(self):
        if self.encoding_type == PositionalEncodingType.Learned:
            self.positional_encoding.data.normal_(0.0, 1 / self.hidden_size)

    def forward(self, X):
        if self.encoding_type.is_added_to_input():
            return X + self.positional_encodings_like(X)
        else:
            return X

    def positional_encodings_like(self, X):
        batch_size, width, height, channels = X.shape  # check channel order
        assert width < MAX_WIDTH_HEIGHT and height < MAX_WIDTH_HEIGHT
        return self.positional_encoding[:width, :height, :].unsqueeze(0)  # unsqueeze the batch size

    def relative_encodings(self, indices):
        assert self.encoding_type.is_relative()
        pass  # TODO
