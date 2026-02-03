# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
import os
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import multiprocessing
import time
import shutil
import itertools
from datetime import datetime
from utils import parse_yaml

parser = ArgumentParser()
parser.add_argument('--config', default='exp_config.yaml')
parser.add_argument('--overwrite', type=str, default='{}')
parser.add_argument('--force_compute', type=str, default='False')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])


import torch
from asv_training.train_asv import train_asv_eval





if __name__ == '__main__':
    multiprocessing.set_start_method("fork",force=True)

    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    eval_data_dir = params['data_dir']
    anon_suffix = params['anon_data_suffix']

    model_dir = params['asv']['training']['model_dir']
    asv_train_params = params['asv']['training']
    if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
        start_time = time.time()
        print('====================')
        print('Perform ASV training')
        print('====================')
        if args.force_compute.lower() == "true":
            shutil.rmtree(model_dir, ignore_errors=True)
        train_asv_eval(train_params=asv_train_params, output_dir=model_dir)
        print("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
        #model_dir = scan_checkpoint(model_dir, 'CKPT')
        shutil.copy(asv_train_params['train_config'], model_dir)
