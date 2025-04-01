import argparse
import numpy as np
import random
import torch
import yaml
# import sys
# sys.path.append('/home/wang')

# from lib.util import get_logger_simple
# from lib.util import get_logger_simple
from lib1.util import get_logger_simple
# from lib.util import get_logger_simple


from layers.trainer import Runner


def main(args):
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
        config['logger'] = logger
        config['data']['dataset_filename'] = f'../data/MILAN/Milan_{args.feature}_{args.num_cells}_10min.csv'
        config['model']['feature'] = args.feature
        config['model']['num_nodes'] = args.num_cells

        if config['train']['is_training']:
            total_iter_times = config['train']['itr']
            for iter_time in range(total_iter_times):
                runner = Runner(iter_time, **config)
                logger.info(f'>>>>>>>start training - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.train()

                logger.info(f'>>>>>>>testing - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.test(load_models=False)

                if config['train']['do_predict']:
                    logger.info(f'>>>>>>>predicting - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.predict(load_models=True)

                torch.cuda.empty_cache()

        else:
            iter_time = 0
            runner = Runner(iter_time, **config)
            if config['train']['do_predict']:
                logger.info(f'>>>>>>>predicting>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.predict(load_models=True)

            else:
                logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.test(load_models=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # generate parser and args
    parser = argparse.ArgumentParser('Baseline-PatchTST')
    parser.add_argument('--dataset', type=str, default='MILAN', choices=['MILAN', 'TRENTINO'], help='dataset')
    parser.add_argument('--num_cells', type=int, default=100, choices=[100, 200, 300, 400, 500, 600, 700, 800, 900,
                                                                       1000], help='Num of Cells')
    parser.add_argument('--feature', type=str, default='Internet', choices=['SMS-in', 'SMS-out', 'Call-in', 'Call-out',
                                                                             'Internet'], help='feature')
    parser.add_argument('--dim', type=str, default='time_loc', choices=['time', 'time_loc', 'time_feat', 'time_loc_feat'],
                        help='the dimension of data')
    args = parser.parse_args()
    args.config_filename = f'../configs/TiDE/TiDE_{args.feature}.yaml'
    logger = get_logger_simple('log', f'TiDE_{args.dataset}_{args.feature}')
    print(f'Baseline:TiDE\t-Dataset:{args.feature}\t-num of cells:{args.num_cells}')

    main(args)
