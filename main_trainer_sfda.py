import argparse
import os
import shutil
import time

from trainers import pmt_Trainer, pmt_Test, deeplab_pmt_Trainer, brats_pmt_Trainer
import json
import glob
import itertools
from options import get_options
import pdb
import sys
import logging
import yaml

#CUDA_VISIBLE_DEVICES
def ensure_dirs(opt):
    '''
    checkpoint dirs - experiment folder of XX dataset XX direction 
    checkpoint dirs - folder for current experiment: {TIME}_{EXPERIMENT_NAME}/
    logger init
    files copy
    '''

    ## 1. checkpoint dirs - experiment folder of XX dataset XX direction 
    if not os.path.exists(opt['checkpoints_dir']):
        os.makedirs(opt['checkpoints_dir'])
    
    ## 2. checkpoint dirs - folder for current experiment: {TIME}_{EXPERIMENT_NAME}/
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if opt['dev']: 
        exp_name = 'dev'
    
    elif opt['config_file'].startswith('configs/test_target_adapt_pmt'):        # test time : adding the best result and validating filder's name to the exp_name
        print(opt['source_model_path'])
        best_result = opt['source_model_path'].split('/')[-1].split('_')[-1][:6]
        load_name =  opt['source_model_path'].split('/')[-3]
        exp_name = f'T{curr_time}_LOADING_{load_name}_RESULT_{best_result}'
        if opt['save']:                                                                      # for visualization use, default is False
            exp_name = f'VISUALUSE_{exp_name}'
    else: 
        exp_name = f'T{curr_time}_ft{opt["train_type"]}_{opt["arch"]}_{opt["pmt_type"]}_w{opt["w1"]}{opt["w2"]}{opt["w3"]}_{opt["note"]}'
        
    opt['checkpoint_dir'] = os.path.join(opt['checkpoints_dir'], exp_name)
    checkpoint_dir = opt['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

        # save current json files into config.json
        with open(os.path.join(checkpoint_dir,'config.json'),'w') as f:
            json.dump(opt, f)

    ## 3. logger initialize
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(f'{checkpoint_dir}/train.log'), logging.StreamHandler(sys.stdout)])
    logging.info(str(opt))
   
    # ## 4. copy files into source_codes folder
    if not os.path.exists(os.path.join(checkpoint_dir,'console_logs')):
        os.makedirs(os.path.join(checkpoint_dir,'console_logs'))

    # if not os.path.exists(os.path.join(checkpoint_dir, 'tf_logs')):
    #     os.makedirs(os.path.join(checkpoint_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoint_dir, 'saved_models'))

    # if not os.path.exists(os.path.join(checkpoint_dir, 'visuals')):
    #     os.makedirs(os.path.join(checkpoint_dir, 'visuals'))
        
    # if not os.path.exists(os.path.join(checkpoint_dir, 'source_codes')):
    #     os.makedirs(os.path.join(checkpoint_dir, 'source_codes'))
        
    #     source_folders = ['.']
    #     sources_to_save = list(itertools.chain.from_iterable(
    #         [glob.glob(f'{folder}/*.py') for folder in source_folders]))
    #     sources_to_save.extend(['./dataloaders', './models','./losses','./trainers','./utils', './configs', './options'])
    #     for source_file in sources_to_save:
    #         if os.path.isfile(source_file):
    #             shutil.copy(source_file,os.path.join(checkpoint_dir, 'source_codes'))
    #         if os.path.isdir(source_file):
    #             if os.path.exists(os.path.join(checkpoint_dir, 'source_codes', source_file)):
    #                 os.removedirs(os.path.join(checkpoint_dir, 'source_codes', source_file))
    #             shutil.copytree(source_file,os.path.join(checkpoint_dir, 'source_codes', source_file),ignore=shutil.ignore_patterns('__pycache__'))
                
if __name__ == '__main__':
    ## param init
    parser = argparse.ArgumentParser(description='Adapt Source Model on Target Images')
    opt = get_options(parser)
    ensure_dirs(opt)

    ## config file and trainer init
    if opt['config_file'].startswith('configs/train_target_adapt_pmt'):
        print('Using pmt prompt Trainer')
        if opt['arch'] in ('Pmt_DeepLab', 'DeepLab'):
            trainer = deeplab_pmt_Trainer(opt)
        # elif opt['dataset_name'] == 'brats':
        #     trainer = brats_pmt_Trainer(opt)
        else:
            trainer = pmt_Trainer(opt)
    elif opt['config_file'].startswith('configs/test_target_adapt_pmt'):
        print('Using pmt prompt test')
        trainer = pmt_Test(opt)

    ## start training
    best_model_path = trainer.launch()
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    ## if want to test at the end of training phase (default on deeplab ct2mr abdominal)
    # opt_test = {}
    # opt_test['config_file'] = 'configs/test_target_adapt_pmt_revise.yaml'
    # opt_test['gpu_id'] = 0
    # opt_test['dev'] = False
    
    # with open(opt_test['config_file']) as f:
    #     config = yaml.safe_load(f)
        
    # opt_test.update(config)
    # opt_test["gpu_id"] = "cuda:%s"%opt_test["gpu_id"]
    # opt_test['checkpoints_dir'] = os.path.join(opt_test['save_root'], opt_test['experiment_name'])
    # opt_test['img_size'] = tuple(opt_test['img_size'])

    # opt_test['source_model_path'] = best_model_path
    # opt_test['ori_model_path'] = opt['bn_align_model']

    # ensure_dirs(opt_test)
    
    # trainer_test = pmt_Test(opt_test)
    # trainer_test.launch()

    

