import torch
import torch.nn.functional as F
import numpy as np
from skimage import transform
from utils.helper import gaussian_kernel3D
import SimpleITK as sitk


def normalization(data, out_min=0, out_max=1, clip=True, clip_percent=0.5):
    """Normalize the input data to the given range
    Args:
        data: input image array
        out_min: minimum of the output
        out_max: maximum of the output
        clip: whether clip the input before normalization
        clip_percent: clip percentage for lower bound, only works when clip=True
    """
    if not clip:
        data_min = np.min(data)
        data_max = np.max(data)
    else:
        # Clip by (clip_percent) and (100-clip_percent) percentage of the input range
        array = data.flatten()
        totalsize = np.size(array)
        lower_bound = int(clip_percent / 100 * totalsize)
        upper_bound = int((100-clip_percent) / 100 * totalsize)
        data_min = array[np.argpartition(array, lower_bound)[lower_bound]]
        data_max = array[np.argpartition(array, upper_bound)[upper_bound]]
        image = np.clip(data, data_min, data_max)

    out = (image - data_min) / (data_max - data_min) * (out_max - out_min) + out_min
    return out


def resize(data, output_size):
    """Resize the input data to the given size
    Args:
        data: input image array
        output_size: output size
    """
    if list(data.shape) != list(output_size):
        data = transform.resize(data, output_size)

    return data


def make_patches(volume, patch_size=[64, 96, 64], patch_stride=24):
    """Extract patches from the input volume
    Args:
        volume: tensor, (B, C, x, z, y), batch and channel will be added if missing
        patch_size: patch size, int or list
        patch_stride: patch stride, int or list

    Returns:
        patches: tensor, (B * patch_num, C, *patch_size). All patches belonging to the same image will be adjacent.
        Reduce any dimension of patch_size that equals 1.
    """
    # Add batch and channel for tensor
    while len(volume.shape) < 5:
        volume = volume.unsqueeze(0)
    # volume.shape, [B, C, *data_size]

    # patching: [patch_size|patch_stride|patch_stride|...|patch_stride]
    pad_residue = [(volume.shape[i + 2] - patch_size[i]) % patch_stride[i] for i in range(3)]
    pad = [0 if pad_residue[i] == 0 else patch_stride[i] - pad_residue[i] for i in range(3)]
    volume = F.pad(volume, (pad[2]//2, (pad[2] + 1)//2, pad[1]//2, (pad[1] + 1)//2, pad[0]//2, (pad[0] + 1)//2),
                   mode='replicate')

    # extract patches using unfold
    # for example, x.shape (m,n,h) --> x.unfold(dimension=1,size,step=size)  --> (m,n/size,h,size)
    patches = volume.unfold(2, patch_size[0], patch_stride[0]).unfold(3, patch_size[1], patch_stride[1]).\
        unfold(4, patch_size[2], patch_stride[2])
    # (B, C, patch_num_d, patch_num_h, patch_num_w, patch_size[0], patch_size[1], patch_size[2])

    # By default, patches belonging to the same image will be adjacent (no need to permute)
    # If calling permute to move batch dimension after patch_num dimensions, patches coming from the same location
    # across all phase images will be stacked together
    # if adjust_order:
    #     patches = patches.permute(1, 2, 3, 4, 0, 5, 6, 7)

    patches = patches.contiguous().view(-1, volume.shape[1], *patch_size)
    # (B * patch_num_d * patch_num_h * patch_num_w, C, patch_size[0], patch_size[1], patch_size[2])

    patches = patches.squeeze(dim=(2, 3, 4))
    return patches


def combine_patches(patches, output_size=[224, 96, 224], patch_size=[64, 96, 64], patch_stride=24):
    """Combine patches back to volume
        Input (B * patch_num, C, *patch_size) and output (B, C, *output_size) are both tensor.
        If stride is larger than patch size during extraction, the output_size and stride for combination should also be
        modified, for example:
            patches = make_patches(volume, patch_size=[224, 1, 224], patch_stride=2), with volume size (224, 96, 224)
            combined = combine_patches(patches, output_size=[224, 49, 224], patch_size=[224, 1, 224], patch_stride=1)
    """
    # F.fold only support 2D image, need to call F.fold twice to achieve 3D patch combination
    # Add dimensions with patch_size equal to 1
    for i in range(3):
        if patch_size[i] == 1:
            patches = patches.unsqueeze(i + 2)

    # determine the required patch numbers along each dimension to obtain the output volume
    dim_patch_num = [int(np.ceil((output_size[i] - patch_size[i]) / patch_stride[i]) + 1) for i in range(3)]
    assert patches.shape[0] % np.prod(dim_patch_num) == 0, 'Incorrect patch number.'

    # determine the extra paddings for output by directly combining all patches
    extended_output_size = [patch_size[i] + (dim_patch_num[i] - 1) * patch_stride[i] for i in range(3)]
    pad = [extended_output_size[i] - output_size[i] for i in range(3)]

    # Use 'gaussian_merge' to avoid the artifacts in patch boundary
    sigma = [patch_size[i]//6 if patch_size[i]//6 > 1 else 1 for i in range(3)]
    gauss_map = torch.from_numpy(gaussian_kernel3D(patch_size, sigma=sigma)).type(patches.dtype).to(patches.device)
    gauss_map = gauss_map.unsqueeze(0).unsqueeze(0)   # (1, 1, patch_size[0], patch_size[1], patch_size[2])

    # patches.shape: (B * patch_num, C, patch_size[0], patch_size[1], patch_size[2]), patch_num = np.prod(dim_patch_num)
    channels = patches.shape[1]
    patches = patches * gauss_map
    # need to combine gaussian maps to normalize the sum at overlapping region, regard each gaussian map as a patch
    weights = gauss_map.repeat(patches.shape[0], channels, 1, 1, 1)

    patches = patches.view(-1, channels, dim_patch_num[0], dim_patch_num[1], dim_patch_num[2], patch_size[0],
                           patch_size[1], patch_size[2])
    # (B, C, dim_patch_num[0], dim_patch_num[1], dim_patch_num[2], patch_size[0], patch_size[1], patch_size[2])

    patches = patches.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, patch_size[0], dim_patch_num[0], patch_size[1], patch_size[2], dim_patch_num[1], dim_patch_num[2])

    patches = patches.contiguous().view(-1, channels * patch_size[0] * dim_patch_num[0] * patch_size[1] * patch_size[2],
                                        dim_patch_num[1] * dim_patch_num[2])
    # (B, C * patch_size[0] * dim_patch_num[0] * patch_size[1]* patch_size[2], dim_patch_num[1] * dim_patch_num[2])

    # F.fold takes input with shape (ch*patch_size, patch_num) --> (ch, *output_size)
    patches = F.fold(patches, output_size=extended_output_size[1:], kernel_size=patch_size[1:], stride=patch_stride[1:])
    # (B, C * patch_size[0] * dim_patch_num[0], extended_output_size[1], extended_output_size[2])

    patches = patches.view(-1, channels * patch_size[0], dim_patch_num[0] * extended_output_size[1] * extended_output_size[2])
    # (B, C * patch_size[0], dim_patch_num[0] * extended_output_size[1] * extended_output_size[2])

    patches = F.fold(patches, output_size=(extended_output_size[0], extended_output_size[1] * extended_output_size[2]),
                     kernel_size=(patch_size[0], 1), stride=(patch_stride[0], 1))
    # (B, C, extended_output_size[0], extended_output_size[1] * extended_output_size[2])

    patches = patches.view(-1, channels, *extended_output_size)
    # (B, C, *extended_output_size)

    # construct weights array to achieve gaussian merge, the same process with patch combination
    weights = weights.view(-1, channels, dim_patch_num[0], dim_patch_num[1], dim_patch_num[2], patch_size[0],
                           patch_size[1], patch_size[2])
    weights = weights.permute(0, 1, 5, 2, 6, 7, 3, 4)
    weights = weights.contiguous().view(-1, channels * patch_size[0] * dim_patch_num[0] * patch_size[1] * patch_size[2],
                                        dim_patch_num[1] * dim_patch_num[2])
    weights = F.fold(weights, output_size=extended_output_size[1:], kernel_size=patch_size[1:], stride=patch_stride[1:])
    weights = weights.view(-1, channels * patch_size[0], dim_patch_num[0] * extended_output_size[1] * extended_output_size[2])
    weights = F.fold(weights, output_size=(extended_output_size[0], extended_output_size[1] * extended_output_size[2]),
                     kernel_size=(patch_size[0], 1), stride=(patch_stride[0], 1))
    weights = weights.view(-1, channels, *extended_output_size)

    # normalization
    patches = patches / weights
    patches = patches[:, :, pad[0] // 2: pad[0] // 2 + output_size[0], pad[1] // 2: pad[1] // 2 + output_size[1],
              pad[2] // 2: pad[2] // 2 + output_size[2]]
    # (B, C, *output_size)

    return patches


def read_image(image_path, norm=True):
    """
    Args:
        image_path: path of the image
        norm: whether perform 0-1 normalization

    Returns:
         tensor
    """
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
    if norm:
        image_array = normalization(image_array, out_min=0, out_max=1, clip=True, clip_percent=0.5)

    tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
    return tensor


# Define a custom collate function to swap the batch and channel batches returned by the DataLoader
def swap_batch_channel_collate_fn(batch):
    # Initialize dictionary to hold processed batch
    processed_batch = {}

    # Extract data and targets
    input_batch = torch.stack([item['input'] for item in batch])  # (batch_size, channels, ...)
    target_batch = torch.stack([item['target'] for item in batch])

    # Rearrange dimensions: change from (batch_size, channels, ...) to (channels, batch_size, ...)
    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)

    processed_batch['input'] = input_batch
    processed_batch['target'] = target_batch

    return processed_batch
