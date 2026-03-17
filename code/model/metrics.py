import torch
import torch.nn as nn
import torch.nn.functional as F


##################
# Structural similarity -- SSIM
# https://github.com/VainF/pytorch-msssim
##################
def _gaussian_1d(size=11, sigma=1.5):
    """Create 1-D gaussian kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, window):
    """ Create gaussian window and blur input using multiple convolutions with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel, [C=1, 1, 1, 1, win_size]
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in window.shape[1:-1]]), window.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):  # [h,d,w]
        if s >= window.shape[-1]:
            # input tensor (batch, in_channels, iD, iH, iW)
            # kernel weights ((out_channels, in_channels/group, kD, kH , kW)
            out = conv(out, weight=window.transpose(2 + i, -1), stride=1, padding=0, groups=C)

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel, [C=1, 1, 1, 1, win_size]
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: ssim results.
    """
    # X.shape: batch, channel, [depth,] height, width
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # flatten from start_dim=2 to end_dim=-1, which will only keep [batch, channel, 1] dimension
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return ssim_per_channel


def calculate_ssim(X, Y, data_range=1, size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03),
         nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of predicted images, (N,C,D,H,W)
        Y (torch.Tensor): a batch of target images, (N,C,D,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if X.shape[0] > Y.shape[0]:
        Y = Y.repeat([X.shape[0] // Y.shape[0]] + [1] * (len(Y.shape) - 1))
    elif X.shape[0] < Y.shape[0]:
        X = X.repeat([Y.shape[0] // X.shape[0]] + [1] * (len(X.shape) - 1))

    if not X.shape == Y.shape:
        raise ValueError(f"After broadcasting, input images should have the shape, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _gaussian_1d(win_size, win_sigma)  # [1, 1, win_size]
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))  # [C=1, 1, 1, 1, win_size]

    ssim_per_channel = _ssim(X, Y, data_range=data_range, win=win, K=K)  # [batch, channel]
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


##################
# Peak signal-to-noise ratio -- PSNR
# PSNR = 20 * log10(Max / RMSE)
##################
def calculate_psnr(X, Y, data_range=1.0, size_average=True):
    """ interface of psnr
    Args:
        X (torch.Tensor): a batch of predicted images, (N,C,D,H,W)
        Y (torch.Tensor): a batch of target images, (N,C,D,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if True, averaged as a scalar, else, return psnr for every batch
    Returns:
        torch.Tensor: psnr results
    """
    if X.shape[0] > Y.shape[0]:
        Y = Y.repeat([X.shape[0] // Y.shape[0]] + [1] * (len(Y.shape) - 1))
    elif X.shape[0] < Y.shape[0]:
        X = X.repeat([Y.shape[0] // X.shape[0]] + [1] * (len(X.shape) - 1))

    if not X.shape == Y.shape:
        raise ValueError(f"After broadcasting, input images should have the shape, but got {X.shape} and {Y.shape}.")

    mse = F.mse_loss(X, Y, reduction='none').mean(dim=tuple(range(2, len(X.shape))))   # [batch, channel]
    if torch.any(mse <= 0):
        return float('inf')

    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))

    if size_average:
        return psnr.mean()      # scalar
    else:
        return psnr.mean(1)    # [batch, ]

