import os
import random
import numpy as np
import torch
import SimpleITK as sitk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     # For multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
def save_model(model, epoch, ckpt_dir, save_interval, save_best=False, loss=0, best_loss=float("Inf")):
    # save model
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ckpt_epoch{epoch}.pth'))
        print(f'Checkpoint {epoch} saved !')

    # save best model
    if save_best and loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), os.path.join(ckpt_dir, r'ckpt_best.pth'))
        with open(os.path.join(ckpt_dir, 'best_loss.txt'), 'a') as f:
            f.write(f'epoch: {epoch} --- loss: {best_loss} \n')

    return best_loss


def gaussian_kernel2D(s=(64, 64), sigma=16):
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    x0 = s[0]//2
    y0 = s[1]//2
    x = np.arange(0, s[0], dtype=float)
    y = np.arange(0, s[1], dtype=float)[:, np.newaxis]
    x -= x0
    y -= y0

    kernel = np.exp(- x ** 2 / (2 * sigma[0] ** 2) - y ** 2 / (2 * sigma[1] ** 2))
    kernel /= np.max(kernel)
    return kernel


def gaussian_kernel3D(s=(64, 64, 64), sigma=16):
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)

    x0 = s[0]//2
    y0 = s[1]//2
    z0 = s[2]//2
    x = np.arange(0, s[0], dtype=float)
    y = np.arange(0, s[1], dtype=float)[:, np.newaxis]
    z = np.arange(0, s[2], dtype=float)[:, np.newaxis, np.newaxis]
    x -= x0
    y -= y0
    z -= z0

    kernel = np.exp(- x ** 2 / (2 * sigma[0] ** 2) - y ** 2 / (2 * sigma[1] ** 2) - z ** 2 / (2 * sigma[2] ** 2))
    kernel /= np.max(kernel)
    return kernel


def save_as_itk(data, save_path, spacing=(2, 2, 2), origin=(-223, -95, -223),
                direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), isVector=False):
    image = sitk.GetImageFromArray(data, isVector=isVector)
    image_size = image.GetSize()

    # default spacing and direction values
    dim = len(image_size)
    if dim == 2 and spacing is None:
        spacing = (1, 1)
    elif dim == 3 and spacing is None:
        spacing = (1, 1, 1)

    if origin is None:
        origin = [-(image_size[i] - 1)/2*spacing[i] for i in range(len(image_size))]

    if dim == 2 and direction is None:
        direction = (1.0, 0.0, 0.0, 1.0)
    elif dim == 3 and direction is None:
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, save_path)
