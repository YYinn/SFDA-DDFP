import argparse
import os
import shutil
import time
from trainers import SourceDomainTrainer,SourceDomainTest
import json
import glob
import itertools
import logging
from options import get_options
import sys

#CUDA_VISIBLE_DEVICES

def ensure_dirs(opt):
    '''
    same in main_trainer_sfda.py
    '''
    if not os.path.exists(opt['checkpoints_dir']):
        os.makedirs(opt['checkpoints_dir'])
        
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if opt['dev']:
        exp_name = 'dev'
    elif  opt['config_file'].startswith('configs/test_source_seg'):
        best_result = opt['source_model_path'].split('/')[-1].split('_')[-1][:6]
        load_name =  opt['source_model_path'].split('/')[-3]
        exp_name = f'T{curr_time}_LOADING_{load_name}_RESULT_{best_result}'
    else:
        exp_name = 'T{}_{}'.format(curr_time, opt['note'])
    

    opt['checkpoint_dir'] = os.path.join(opt['checkpoints_dir'],exp_name)
    checkpoint_dir = opt['checkpoint_dir']
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir,'config.json'),'w') as f:
            json.dump(opt,f)


    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(f'{checkpoint_dir}/train.log'), logging.StreamHandler(sys.stdout)])
    logging.info(str(opt))
    
    if not os.path.exists(os.path.join(checkpoint_dir,'console_logs')):
        os.makedirs(os.path.join(checkpoint_dir,'console_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'tf_logs')):
        os.makedirs(os.path.join(checkpoint_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoint_dir, 'saved_models'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoint_dir, 'visuals'))
        
    if not os.path.exists(os.path.join(checkpoint_dir, 'source_codes')):
        os.makedirs(os.path.join(checkpoint_dir, 'source_codes'))
        
        source_folders = ['.']
        sources_to_save = list(itertools.chain.from_iterable(
            [glob.glob(f'{folder}/*.py') for folder in source_folders]))
        sources_to_save.extend(['./dataloaders', './models','./losses','./trainers','./utils'])
        for source_file in sources_to_save:
            if os.path.isfile(source_file):
                shutil.copy(source_file,os.path.join(checkpoint_dir, 'source_codes'))
            if os.path.isdir(source_file):
                if os.path.exists(os.path.join(checkpoint_dir, 'source_codes', source_file)):
                    os.removedirs(os.path.join(checkpoint_dir, 'source_codes', source_file))
                shutil.copytree(source_file,os.path.join(checkpoint_dir, 'source_codes', source_file),ignore=shutil.ignore_patterns('__pycache__'))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentor on Source Images')
    opt = get_options(parser)
    ensure_dirs(opt)

    if opt['config_file'].startswith('configs/train_source_seg'): 
        print('======= Using train Trainer')
        trainer = SourceDomainTrainer(opt)
    elif opt['config_file'].startswith('configs/test_source_seg'): 
        print('======= Using test Trainer')
        trainer = SourceDomainTest(opt)

    trainer.launch()