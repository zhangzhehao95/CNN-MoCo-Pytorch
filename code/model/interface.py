import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.processing import read_image, make_patches, combine_patches
from dataset.dataset_generator import get_dataloader
from model.metrics import calculate_ssim, calculate_psnr
from utils.helper import save_model, save_as_itk
from utils.lr_scheduler import LR_scheduler
from utils.stop_criteria import StopCriteria_NoImprove

import tqdm
from torchinfo import summary


def train(model, cf):
    print('\n > Creating training dataset...')
    train_loader = get_dataloader('train', cf)
    # training batch number
    n_train = len(train_loader)

    # Load validation data
    if cf['valid']:
        print('\n > Creating validation dataset...')
        val_loader = get_dataloader('valid', cf)
    # visualization of network
    input_size = [s for s in cf['patch_size'] if s > 1]
    summary(model, input_size=(1, 1, *input_size), depth=6)

    # Checkpoints
    ckpt_dir = os.path.join(cf['output_folder'], 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Resume previous training
    if cf['resume']:
        model.load_state_dict(torch.load(cf['resume_path'], map_location=cf['device'], weights_only=True))
        print("\n > Resume training with pre-trained weights: " + cf['resume_path'])

    # Loss function, optimizer, scheduler
    if cf['loss'] == 'L1':
        loss_func = nn.L1Loss()
    elif cf['loss'] == 'L2':
        loss_func = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cf['init_lr'], weight_decay=cf['weight_decay'])
    scheduler = LR_scheduler(optimizer, scheduler_type=cf['lr_scheduler']['type'],
                             gamma=cf['lr_scheduler']['factor'], tol_epochs=cf['epoch_num'],
                             spe_arg=cf['lr_scheduler']['spe_arg'])

    # Choose stopping criteria
    if cf['early_stopping']['type'] == 'loss_improve':
        stop_criterion = StopCriteria_NoImprove(query_len=cf['early_stopping']['patience'],
                                                num_min_epoch=cf['early_stopping']['min_epoch'],
                                                min_improve=cf['early_stopping']['min_improve'])
    else:
        stop_criterion = None

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(cf['output_folder'], 'tensorboard_' + cf['timestamp']))

    best_loss = float("Inf")
    global_step = 0
    # Epoch iterations
    for epoch in range(cf['epoch_num']):
        model.train()
        ave_train_loss = 0  # average total loss over each training epoch

        # Iterations in one epoch
        pbar = tqdm.tqdm(train_loader, total=n_train, desc=f"Epoch {epoch + 1}/{cf['epoch_num']}", unit='batch')
        for train_pair in pbar:
            x = train_pair['input'].to(device=cf['device'])
            y = train_pair['target'].to(device=cf['device'])

            y_predict = model(x)
            loss = loss_func(y_predict, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            ave_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            tb_writer.add_scalar('Train/Loss', loss.item(), global_step)

        ave_train_loss /= n_train
        tb_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
        tb_writer.add_scalar('Train/Epoch_ave_loss', ave_train_loss, epoch + 1)
        scheduler.step()

        # Validation after each epoch
        if cf['valid']:
            # no need to save image results
            ave_loss, ave_psnr, ave_ssim = evaluate(model, val_loader, loss_func, cf, results_save_path=None)

            print(f'Validation loss: {ave_loss}, validation PSNR: {ave_psnr}, validation SSIM: {ave_ssim}')
            tb_writer.add_scalar('Validation/Loss', ave_loss, epoch + 1)
            tb_writer.add_scalar('Validation/PSNR', ave_psnr, epoch + 1)
            tb_writer.add_scalar('Validation/SSIM', ave_ssim, epoch + 1)

        # Use validation loss as current loss if possible
        cur_loss = ave_loss if cf['valid'] else ave_train_loss
        # stop criteria
        if stop_criterion is not None:
            stop_criterion.add(cur_loss)
            if stop_criterion.stop():
                print('\n > Early stopping!')
                break

        # save weights
        if cf['ckpt_enabled']:
            best_loss = save_model(model, epoch + 1, ckpt_dir, cf['save_interval'], save_best=cf['save_best'],
                                   loss=cur_loss, best_loss=best_loss)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'final_weights.pth'))
    print('\n > Training finished.')


# produce prediction and perform quantitative evaluation
def evaluate(model, dataloader, loss_func, cf, results_save_path=None):
    model.eval()

    n_batches = len(dataloader)
    ave_loss = 0
    psnr_list = []
    ssim_list = []
    patch_count = 0    # record for processed patches
    volume_index = 0   # record for processed volumes

    for pair in dataloader:
        x = pair['input'].to(device=cf['device'])
        y = pair['target'].to(device=cf['device'])
        # [B=1, C=1, *patch_size]

        with torch.no_grad():
            y_predict = model(x)
            ave_loss += loss_func(y_predict, y).item()

            if patch_count == 0:
                patch_input = x
                patch_target = y
                patch_predict = y_predict
                # (1, C=1, *patch_size)
            else:
                patch_input = torch.cat((patch_input, x), 0)
                patch_target = torch.cat((patch_target, y), 0)
                patch_predict = torch.cat((patch_predict, y_predict), 0)
                # (patch_num, 1, *patch_size)

            patch_count += 1

            if patch_count == cf['patch_num_per_volume']:
                case = volume_index // cf['phase_num']
                p = volume_index % cf['phase_num']
                suffix = f"{case}_phase{p}.mha"
                volume_index += 1
                patch_count = 0

                image_input = combine_patches(patch_input, output_size=cf['data_size'], patch_size=cf['patch_size'],
                                              patch_stride=cf['patch_stride'])
                image_target = combine_patches(patch_target, output_size=cf['data_size'], patch_size=cf['patch_size'],
                                               patch_stride=cf['patch_stride'])
                image_predict = combine_patches(patch_predict, output_size=cf['data_size'], patch_size=cf['patch_size'],
                                                patch_stride=cf['patch_stride'])
                # (B=1, C=1, *data_size)

                psnr_list.append(calculate_psnr(image_predict, image_target).item())
                ssim_list.append(calculate_ssim(image_predict, image_target).item())

                if results_save_path is not None:
                    if cf['save_input_mha']:
                        save_as_itk(image_input.squeeze().cpu().numpy(),
                                    os.path.join(results_save_path, 'Input' + suffix),
                                    spacing=cf['spacing'], origin=cf['origin'], direction=cf['direction'])

                    save_as_itk(image_predict.squeeze().cpu().numpy(),
                                os.path.join(results_save_path, 'Output' + suffix),
                                spacing=cf['spacing'], origin=cf['origin'], direction=cf['direction'])

    ave_loss /= n_batches           # patch level
    ave_psnr = np.mean(psnr_list)   # image level
    ave_ssim = np.mean(ssim_list)   # image level

    return ave_loss, ave_psnr, ave_ssim


# test on new input, perform quality evaluation with targets, save test results
def test(model, cf):
    load_weights_path = os.path.join(cf['output_folder'], 'checkpoints', cf['pretrained_ckpt_file'])
    model.load_state_dict(torch.load(load_weights_path, map_location=cf['device'], weights_only=True))
    print(f"\n > Loading pre-trained weights: {load_weights_path}")

    results_save_path = os.path.join(cf['output_folder'], 'test_results', 'test_data')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    print('\n > Creating testing dataset...')
    test_loader = get_dataloader('test', cf)

    # Loss function
    if cf['loss'] == 'L1':
        loss_func = nn.L1Loss()
    elif cf['loss'] == 'L2':
        loss_func = nn.MSELoss()

    ave_loss, ave_psnr, ave_ssim = evaluate(model, test_loader, loss_func, cf, results_save_path=results_save_path)
    print(f'Testing loss: {ave_loss}, testing PSNR: {ave_psnr}, testing SSIM: {ave_ssim}')

    print('\n > Testing finished.')


# Infer on new input, no target/ground truth available, thus no evaluation
def predict(model, cf):
    load_weights_path = os.path.join(cf['output_folder'], 'checkpoints', cf['pretrained_ckpt_file'])
    model.load_state_dict(torch.load(load_weights_path, map_location=cf['device'], weights_only=True))
    print(f"\n > Loading pre-trained weights: {load_weights_path}")
    model.eval()

    results_save_path = os.path.join(cf['output_folder'], 'test_results')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    predict_data_path = cf['predict_dir']

    for case_dir in os.listdir(predict_data_path):
        case_path = os.path.join(predict_data_path, case_dir)
        if os.path.isdir(case_path):  # dataset/case_name/{FDK_phase*}.mha
            # Sub-folder for prediction results
            sub_save_path = os.path.join(results_save_path, case_dir)
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)

            for p in range(cf['phase_num']):
                input_file = os.path.join(case_path, f"FDK_phase{p}.mha")  # 4D FDK

                suffix = f"_phase{p}.mha"
                # Load image, normalization, patching
                image_input = read_image(input_file)

                # infer on whole volume at once
                if cf.get('volume_infer', False):
                    x = image_input.unsqueeze(0).unsqueeze(0).to(device=cf['device'])  # [B=1, C=1, *data_size]

                    with torch.no_grad():
                        image_predict = model(x)

                # predict patch by patch
                else:
                    patch_input = make_patches(image_input, cf['patch_size'], cf['patch_stride'])
                    # [patch_num, 1, *patch_size]

                    patch_num = patch_input.shape[0]
                    for i in range(patch_num):
                        x = patch_input[i, ...].unsqueeze(0).to(device=cf['device'])    # [B=1, C=1, *patch_size]

                        with torch.no_grad():
                            y_predict = model(x)
                            if i == 0:
                                patch_predict = y_predict
                            else:
                                patch_predict = torch.cat((patch_predict, y_predict), 0)

                    image_predict = combine_patches(patch_predict, output_size=cf['data_size'],
                                                    patch_size=cf['patch_size'], patch_stride=cf['patch_stride'])

                if cf['save_input_mha']:
                    save_as_itk(image_input.cpu().numpy(), os.path.join(sub_save_path, 'Input' + suffix),
                                spacing=cf['spacing'], origin=cf['origin'], direction=cf['direction'])

                save_as_itk(image_predict.squeeze().cpu().numpy(), os.path.join(sub_save_path, 'Output' + suffix),
                            spacing=cf['spacing'], origin=cf['origin'], direction=cf['direction'])

    print('\n > Prediction finished.')
