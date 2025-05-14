import os
import yaml
import argparse

def get_options(parser):
    '''
    get configs from config file
    '''

    ## Config file
    parser.add_argument("--config_file",type=str, default='configs/train_target_adapt_pmt_revise.yaml')
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dev", default=False, action='store_true')
    parser.add_argument("--note", default='', type=str)

    # ## ablation
    # parser.add_argument("--w1", default=1, type=float)
    # parser.add_argument("--w2", default=1, type=float)
    # parser.add_argument("--w3", default=10, type=float)
    
    opt = vars(parser.parse_args())
    with open(opt['config_file']) as f:
        config = yaml.safe_load(f)
        
    opt.update(config)
    opt["gpu_id"] = "cuda:%s"%opt["gpu_id"]
    opt['checkpoints_dir'] = os.path.join(opt['save_root'],opt['experiment_name'])
    opt['img_size'] = tuple(opt['img_size'])

    return opt