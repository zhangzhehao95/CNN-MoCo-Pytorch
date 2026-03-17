import os
import sys
import random
import torch
import yaml
import shutil

from argparse import ArgumentParser
from datetime import datetime

from utils.logger import Logger
from utils.helper import set_seed
from utils.configuration import configuration

import model.model_factory as model_factory
from model.interface import train, test, predict


def main():
    parser = ArgumentParser(description='Artifact-reduction network for 4D-CBCT')
    parser.add_argument('-c', '--config', type=str, help='path to the config file.', required=True)
    args = parser.parse_args()

    # load the yaml config file
    config_file = args.config
    with open(config_file) as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)

    # prepare output folder
    # outputs of each experiment will be saved into a folder named by 'exp_name' under 'save_dir'
    output_folder = os.path.join(cf['save_dir'], cf['exp_name'])
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cf['output_folder'] = output_folder
    cf['timestamp'] = timestamp

    # enable log recording for training process
    if cf['train']:
        log_file = os.path.join(output_folder, 'logfile_' + timestamp + '.log')
        sys.stdout = Logger(log_file)   # Rewrite the print function
        shutil.copyfile(config_file, os.path.join(output_folder, 'config_' + timestamp + '.yaml'))  # copy config file

    print(' > Configuration file path: ' + config_file)
    # fix random seed for reproducibility
    seed = cf['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    set_seed(seed)
    print('\n > Utilized random seed: ' + str(seed))

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n > Utilized device: ' + str(device))
    cf['device'] = device

    # make necessary modifications on config
    cf = configuration(cf)

    # build model
    model_constructor = getattr(model_factory, cf['model'])
    model = model_constructor(cf).to(device=device)
    print('\n > Building model: ' + cf['model'])

    # training
    if cf['train']:
        print('\n > Start the training process...')
        train(model, cf)

    # testing
    if cf['test']:
        print('\n > Start the testing process...')
        test(model, cf)

    # prediction, where no targets are available
    if cf['predict']:
        print('\n > Start the prediction process...')
        predict(model, cf)


if __name__ == '__main__':
    main()


