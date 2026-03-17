import numpy as np


def configuration(cf):
    """ Make necessary modifications for configuration """
    # patch_size and patch_stride should be a list with the length of 3
    if isinstance(cf['patch_size'], int):
        cf['patch_size'] = [cf['patch_size']] * 3

    if isinstance(cf['patch_stride'], int):
        cf['patch_stride'] = [cf['patch_stride']] * 3

    # calculate how many patches are in one volume
    dim_patch_num = [int(np.ceil((cf['data_size'][i] - cf['patch_size'][i]) / cf['patch_stride'][i]) + 1) for i in range(3)]
    cf['patch_num_per_volume'] = np.prod(dim_patch_num)

    # determine modes for convolution blocks
    mode = ''
    # normalization function
    if cf['normalization'] == 'Batch':
        mode += 'B'
    elif cf['normalization'] == 'Instance':
        mode += 'I'

    # activation function
    if cf['activation'] == 'ReLU':
        mode += 'R'
    elif cf['activation'] == 'LeakyReLU':
        mode += 'L'
    elif cf['activation'] == 'PReLU':
        mode += 'P'
    elif cf['activation'] == 'ELU':
        mode += 'E'

    cf['conv_mode'] = 'C' + mode
    cf['transconv_mode'] = 'T' + mode

    return cf
