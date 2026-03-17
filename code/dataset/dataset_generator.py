import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataset.processing import read_image, make_patches, swap_batch_channel_collate_fn


class ArtifactReductionDataset(Dataset):
    """
        Determine the dataset type based on the existence of pseudo_average_phase0.mha under each case folder.
        For self_contain type dataset:
            Constructed pseudo-average images and 3D_FDK as training pair
            Read 3D_FDK only once
            specific parameter: unbiased_batch
                Whether to force all phases image corresponding to the same high-quality image belong to the same batch
                If True, return (n_phase, *patch_size). Need to always set batch size equal to 1 in DataLoader,
                    after adding batch, also need to swap axes.
                If False, default, return (1, *patch_size).

        For ground_truth type dataset:
            Use 4D_FDK and GT as training pair (simulation)
    """
    def __init__(self, data_path, n_phases=10, patch_size=(64, 96, 64), patch_stride=(40, 1, 40), unbiased_batch=False):
        # Training and validation
        self.input_list = []
        self.target_list = []

        image_pair_num = 0
        for case_dir in os.listdir(data_path):
            case_path = os.path.join(data_path, case_dir)
            if os.path.isdir(case_path):
                # Determine the dataset type by checking the existence of pseudo average phase image
                if os.path.exists(os.path.join(case_path, f"pseudo_average_phase0.mha")):
                    dataset_type = 'self_contain'
                else:
                    dataset_type = 'ground_truth'

                # 'self_contain': dataset/case_name/{pseudo_average_phase*, 3DFDK}.mha
                if dataset_type == 'self_contain':
                    for p in range(n_phases):
                        input_file = os.path.join(case_path, f"pseudo_average_phase{p}.mha")  # 4D FDK
                        # Read MHA image and perform intensity normalization
                        input_image = read_image(input_file)    # (x, z, y)
                        if p == 0:
                            input_batch = input_image.unsqueeze(0).unsqueeze(0)
                        else:
                            input_batch = torch.cat((input_batch, input_image.unsqueeze(0).unsqueeze(0)), 0)
                            # (B=n_phases, 1, x, z, y)

                        image_pair_num += 1

                    target_file = os.path.join(case_path, f"3DFDK.mha")  # 3D FDK
                    out_image = read_image(target_file)

                    input_batch_patches = make_patches(input_batch, patch_size, patch_stride)
                    # patches from the same phase image are adjacent, (patch_num * n_phase, 1, *patch_size)

                    target_batch_patches = make_patches(out_image, patch_size, patch_stride)
                    # (patch_num, 1, *patch_size)
                    patch_num_per_volume = target_batch_patches.shape[0]

                    if unbiased_batch:
                        # All images corresponding to the same high-quality image will be put into one batch
                        for i in range(patch_num_per_volume):
                            same_patch_across_all_phases = range(i, input_batch_patches.shape[0], patch_num_per_volume)
                            self.input_list.append(input_batch_patches[same_patch_across_all_phases, ...].squeeze())
                            # (n_phase, *patch_size), need to swap Batch and Channel after batching.

                            self.target_list.append(target_batch_patches[[i], ...].repeat(
                                [n_phases] + [1] * (len(target_batch_patches.shape) - 1)).squeeze())
                            # (n_phase, *patch_size)
                    else:
                        for i in range(0, input_batch_patches.shape[0]):
                            self.input_list.append(input_batch_patches[i, ...])
                            # (C=1, *patch_size)

                            self.target_list.append(target_batch_patches[i % patch_num_per_volume, ...])
                            # (C=1, *patch_size)

                # 'ground_truth': dataset/case_name/{pseudo_average_phase*, 3DFDK}.mha
                elif dataset_type == 'ground_truth':
                    for p in range(n_phases):
                        input_file = os.path.join(case_path, f"FDK_phase{p}.mha")  # 4D FDK
                        input_image = read_image(input_file)  # (x, z, y)

                        target_file = os.path.join(case_path, f"GT_phase{p}.mha")  # 3D FDK
                        target_image = read_image(target_file)

                        input_patches = make_patches(input_image, patch_size, patch_stride)
                        # (patch_num, 1, *patch_size)
                        self.input_list.extend([input_patches[j, ...] for j in range(input_patches.shape[0])])
                        # list of tensors with shape [C=1, *patch_size]

                        target_patches = make_patches(target_image, patch_size, patch_stride)
                        self.target_list.extend([target_patches[j, ...] for j in range(target_patches.shape[0])])

                        image_pair_num += 1

        self.ids = len(self.input_list)
        print(f"\n > Dataset contains {image_pair_num} pairs of images, with {self.ids} pairs of patches and "
              f"unbiased batching set to be {unbiased_batch}.")

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        input_artifact = self.input_list[idx]
        target_clean = self.target_list[idx]

        return {'input': input_artifact, 'target': target_clean}


def get_dataloader(mode, cf):
    """
        mode: 'train' | 'valid' | 'test'
        Corresponding dataset patch accepts string or list: mode + '_dir'
    """
    # Use fixed batch size of 1 and no-shuffle for non-training mode
    if mode == 'train':
        shuffle = True
        batch = cf['batch_size']
    else:
        shuffle = False
        batch = 1

    dataset_dir = cf[mode + '_dir']

    # unbiased_batch batching mode is only for self-contained training, by binding images corresponding to the
    # same ground truth into the same one batch. Always set batch size to be 1 when unbiased_batch == True,
    # also need to switch batch and channel after dataset loader fetching.
    if cf['unbiased_batch']:
        batch = 1
        collate_fn = swap_batch_channel_collate_fn
    else:
        collate_fn = None

    if isinstance(dataset_dir, list):
        # chain multiple datasets
        for i, i_dir in enumerate(dataset_dir):
            print('\n > Loading dataset: ' + i_dir)
            if i == 0:
                dataset = ArtifactReductionDataset(i_dir, cf['phase_num'], cf['patch_size'], cf['patch_stride'],
                                                   cf['unbiased_batch'])
            else:
                dataset = ConcatDataset([dataset, ArtifactReductionDataset(i_dir, cf['phase_num'], cf['patch_size'],
                                                                           cf['patch_stride'], cf['unbiased_batch'])])

        # create dataset loader
        dataset_loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, collate_fn=collate_fn, num_workers=0,
                                    drop_last=False, worker_init_fn=np.random.seed(0))
    else:
        print('\n > Loading dataset: ' + dataset_dir)
        dataset = ArtifactReductionDataset(dataset_dir, cf['phase_num'], cf['patch_size'], cf['patch_stride'],
                                           cf['unbiased_batch'])
        dataset_loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, collate_fn=collate_fn, num_workers=0,
                                    drop_last=False, worker_init_fn=np.random.seed(0))

    return dataset_loader

