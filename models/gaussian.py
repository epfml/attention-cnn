import torch
import numbers


def gaussian_kernel_2d(mean, std_inv, size):
    """Create a 2D gaussian kernel

    Args:
        mean: center of the gaussian filter (shift from origin)
            (2, ) vector
        std_inv: standard deviation $Sigma^{-1/2}$
            can be a single number, a vector of dimension 2, or a 2x2 matrix
        size: size of the kernel
            pair of integer for width and height
            or single number will be used for both width and height

    Returns:
        A gaussian kernel of shape size.
    """
    if type(mean) is torch.Tensor:
        device = mean.device
    elif type(std_inv) is torch.Tensor:
        device = std_inv.device
    else:
        device = "cpu"

    # repeat the size for width, height if single number
    if isinstance(size, numbers.Number):
        width = height = size
    else:
        width, height = size

    # expand std to (2, 2) matrix
    if isinstance(std_inv, numbers.Number):
        std_inv = torch.tensor([[std_inv, 0], [0, std_inv]], device=device)
    elif std_inv.dim() == 0:
        std_inv = torch.diag(std_inv.repeat(2))
    elif std_inv.dim() == 1:
        assert len(std_inv) == 2
        std_inv = torch.diag(std_inv)

    # Enforce PSD of covariance matrix
    covariance_inv = std_inv.transpose(0, 1) @ std_inv
    covariance_inv = covariance_inv.float()

    # make a grid (width, height, 2)
    X = torch.cat(
        [
            t.unsqueeze(-1)
            for t in reversed(
                torch.meshgrid(
                    [torch.arange(s, device=device) for s in [width, height]]
                )
            )
        ],
        dim=-1,
    )
    X = X.float()

    # center the gaussian in (0, 0) and then shift to mean
    X -= torch.tensor([(width - 1) / 2, (height - 1) / 2], device=device).float()
    X -= mean.float()

    # does not use the normalize constant of gaussian distribution
    Y = torch.exp((-1 / 2) * torch.einsum("xyi,ij,xyj->xy", [X, covariance_inv, X]))

    # normalize
    # TODO could compute the correct normalization (1/2pi det ...)
    # and send warning if there is a significant diff
    # -> part of the gaussian is outside the kernel
    Z = Y / Y.sum()
    return Z
