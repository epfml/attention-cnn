import torch
import torch.nn as nn
from opt_einsum import contract

def dilat(tensor, dilations):
    """
    tensor  (tensor): Tensor to be dilated
    dilations (list): List of dilation factor for each dim,
                      None for no dilation on this dim

    For each dilated dim of size s, it splits this dim into to (s // dil, dil) dimensions
    Otherwise, does nothing to the dim
    """
    assert len(tensor.shape) == len(dilations)
    assert all(
        (dil is None) or (sh % dil == 0) for dil, sh in zip(dilations, tensor.shape)
    ), "dilation should divide dimension"

    new_dim = []
    content_indices = []
    group_indices = []
    for dim, dilat in zip(tensor.shape, dilations):
        if dilat is None:
            new_dim.append(dim)
        else:
            new_dim.extend([dim // dilat, dilat])

    tensor_dilat = tensor.view(*new_dim)
    return tensor_dilat


def dilated_attention(V, Q, K, dilation=1):
    try:
        x_dilation, y_dilation = dilation
    except:
        x_dilation = y_dilation = dilation

    # V shape: B x W x H x n_head x d_v
    # Q shape: B x W x H x n_head x d_k
    # K shape: B x W x H x n_head x d_k
    batch_size, width, height, n_head, d_v = V.shape
    d_k = Q.shape[-1]

    # B x W/dil x dil x H/dil x dil x n_head x d
    K_dilated = dilat(K, [None, x_dilation, y_dilation, None, None])
    Q_dilated = dilat(Q, [None, x_dilation, y_dilation, None, None])
    V_dilated = dilat(V, [None, x_dilation, y_dilation, None, None])

    # b = batch, h = head, d = dimension
    # each pixel can attend pixels in its own group,
    # i.e. all position at a distance multiple of (x_dil, y_dil)
    # blocks are (x_dil X y_dil) rectangle of the image
    # each pixel of block (x,y) in the group (i,j) attend pixel in same group in different blocks (v,w)
    # a dot product is done over the d dimension
    # the head and batch dimension are kept
    attention_coefficients = contract("bxiyjhd,bviwjhd->bxiyjhvw", Q_dilated, K_dilated)
    #attention_coefficients = torch.einsum("bxiyjhd,bviwjhd->bxiyjvwh", [Q_dilated, K_dilated])
    attention_shape = attention_coefficients.size()
    attention_coefficients = attention_coefficients.view(attention_shape[:-2] + (-1,))
    attention_probs = nn.Softmax(dim=-1)(attention_coefficients)
    attention_coefficients = attention_probs.view(attention_shape)



    # the attention_coefficients are used to compute the weighted sum of the values
    # each pixel in block (x,y) and group (i,j) sums the values of
    # the pixes in group (i,j) at any other block position (v,w)
    new_V = contract("bxiyjhvw,bviwjhd->bxiyjhd", attention_coefficients, V_dilated)
    new_V = new_V.contiguous().view(batch_size, width, height, n_head*d_v)
    #print(new_V.shape)
    return new_V, attention_coefficients


if __name__ == "__main__":
    batch_size = 33
    width = 55
    height = 44
    n_head = 7
    d = 6
    dil = 11

    print("batch_size", batch_size)
    print("width", width)
    print("height", height)
    print("n_head", n_head)
    print("d", d)
    print("dil", dil)

    V = torch.rand(batch_size, width, height, n_head, d)
    K = torch.rand(batch_size, width, height, n_head, d)
    Q = torch.rand(batch_size, width, height, n_head, d)

    new_V, attention_coefficients = dilated_attention(V, Q, K, dil)

    print("K:", K.shape)
    print("Q:", Q.shape)
    print("V:", V.shape)
    print("attention_coefficients:", attention_coefficients.shape)
    print("new_V:", new_V.shape)
