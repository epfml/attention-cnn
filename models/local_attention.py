import torch
import torch.nn as nn
from opt_einsum import contract



def get_unfolded(tensor,kernel_size):
    """
    tensor  (tensor): Tensor to be dilated, in shape (batch_size, W, H, nhead*d)
    kernel_size: size of the square to be attended

    output:
        tensor_unf: results in shape (batch, nhead*d, W, H, kernel_size, kernel_size)

    """
    B, W, H, D = tensor.shape
    tensor = tensor.permute(0,3,1,2)
    unf = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=int((kernel_size-1)/2), stride=1)
    tensor_unf = unf(tensor)
    tensor_unf = tensor_unf.view((B, D, W, H , kernel_size, kernel_size))
    return tensor_unf


def local_attention(V, Q, K, kernel_size=5):
    """

    :param V: of shape( batchsize, width, height, numberOfHeads, hidden_dim/numOfHeads)
    :param Q:
    :param K:
    :param kernel_size:
    :return: updated V of shape (batch_size, width, height, hidden_dim)
    """
    # V shape: B x W x H x n_head x d_v
    # Q shape: B x W x H x n_head x d_k
    # K shape: B x W x H x n_head x d_k
    batch_size, width, height, n_head, d_v = V.shape
    V = V.view((batch_size, width, height, -1))
    #Q = Q.view((batch_size, width, height, -1))
    K = K.view((batch_size, width, height, -1))

    d_k = Q.shape[-1]

    K_field = get_unfolded(K,kernel_size).view((batch_size, n_head, d_v, width, height, kernel_size, kernel_size))
    V_field = get_unfolded(V,kernel_size).view((batch_size, n_head, d_v, width, height, kernel_size, kernel_size))
    #Q_field = get_unfolded(Q)

    # b = batch, h = head, d = dimension
    # each pixel can attend pixels in its own group,
    # i.e. all position at a distance multiple of (x_dil, y_dil)
    # blocks are (x_dil X y_dil) rectangle of the image
    # each pixel of block (x,y) in the group (i,j) attend pixel in same group in different blocks (v,w)
    # a dot product is done over the d dimension
    # the head and batch dimension are kept
    attention_coefficients = contract("bwhnd,bndwhxy->bwhnxy", Q, K_field,backend='torch')
    #attention_coefficients = torch.einsum("bxiyjhd,bviwjhd->bxiyjvwh", [Q_dilated, K_dilated])
    attention_shape = attention_coefficients.size()
    attention_coefficients = attention_coefficients.view(attention_shape[:-2] + (-1,))
    attention_probs = nn.Softmax(dim=-1)(attention_coefficients)
    attention_coefficients = attention_probs.view(attention_shape)



    # the attention_coefficients are used to compute the weighted sum of the values
    # each pixel in block (x,y) and group (i,j) sums the values of
    # the pixes in group (i,j) at any other block position (v,w)
    new_V = contract("bwhnxy,bndwhxy->bwhnd", attention_coefficients, V_field,backend='torch')
    #print(new_V.shape)
    new_V = new_V.contiguous().view(batch_size, width, height, n_head* d_v)
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

    new_V, attention_coefficients = local_attention(V, Q, K, 5)

    print("K:", K.shape)
    print("Q:", Q.shape)
    print("V:", V.shape)
    print("attention_coefficients:", attention_coefficients.shape)
    print("new_V:", new_V.shape)