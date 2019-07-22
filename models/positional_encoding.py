from enum import Enum


class PositionalEncodingType(Enum):
    Sinusoid2d = "Sinusoid2d"
    Learned = "Learned"
    Relative = "Relative"
    Nothing = "Nothing"
