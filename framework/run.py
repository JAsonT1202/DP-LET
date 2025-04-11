import argparse
import os
import sys
from lib.util import get_logger_simple
import random
import torch
import numpy as np
import pandas as pd
import yaml
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
        config['model']['dim'] = args.dim

        if config['model']['dim'] == 'time':
            mae_list, mse_list, mape_list = [], [], []
            for i in range(config['model']['num_nodes']):
                config['cell_idx'] = i
                print(f'Univariate time series prediction-time dim-cell{i}')
                if config['is_training']:
                    for iter_time in range(config['train']['itr']):
                        runner = Runner(iter_time, **config)
                        logger.info(f'>>>>>>>start training - cell: {i} - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        runner.train()

                        logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        mae, mse, mape = runner.test(load_models=False)
                        mae_list.append(mae)
                        mse_list.append(mse)
                        mape_list.append(mape)


                        if config['train']['do_predict']:
                            logger.info(f'>>>>>>>predicting - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                            runner.predict(load_models=True)

                else:
                    iter_time = 0
                    runner = Runner(iter_time, **config)
                    logger.info(f'>>>>>>>cell: {i} testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.test(load_models=True)
                    torch.cuda.empty_cache()

            f = open('result.txt', 'a')
            f.write(f'mse:{np.mean(mse_list)}, mae:{np.mean(mae_list)}, mape:{np.mean(mape_list)}')
            f.write('\n')
            f.write('\n')
            f.close()

        elif config['model']['dim'] == 'time_loc':
            if config['is_training']:
                for iter_time in range(config['train']['itr']):
                    runner = Runner(iter_time, **config)
                    logger.info(f'>>>>>>>start training - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.train()

                    logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.test(load_models=False)

                    if config['train']['do_predict']:
                        logger.info(f'>>>>>>>predicting - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        runner.predict(load_models=True)
            else:
                iter_time = 0
                runner = Runner(iter_time, **config)
                logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.test(load_models=True)
                torch.cuda.empty_cache()

        elif config['model']['dim'] == 'time_feat':
            config['data']['dataset_filename'] = f'../data/MILAN/Milan_{args.feature}_{args.num_cells}_10min.csv'
            # df_raw = pd.read_csv(config['data']['dataset_filename'])
            # cols = df_raw.columns
            feat_list = ['SMS-in', 'SMS-out', 'Call-in', 'Call-out', 'Internet']
            df_raw_list = []
            for feat in feat_list:
                df_raw = pd.read_csv(f'../data/MILAN/Milan_{feat}_{args.num_cells}_10min.csv')
                df_raw_list.append(df_raw)
            cols = df_raw.columns

            mae_list, mse_list, mape_list = [], [], []
            for i, col in enumerate(cols[1:]):
                idx_name = pd.date_range(start='2013/11/01 00:00:00', end='2014/01/01 23:50:00', freq='10T')
                df = pd.DataFrame(data=np.zeros(shape=(8928, 5)), index=idx_name, columns=feat_list)

                for idx, feat in enumerate(feat_list):
                    feat_value = df_raw_list[idx][col].values
                    df[feat] = feat_value
                config['cell_idx'] = i
                config['data']['dataset_filename'] = df
                print(f'Multi-variate time series prediction-time feature dim-cell{i}')
                if config['is_training']:
                    for iter_time in range(config['train']['itr']):
                        runner = Runner(iter_time, **config)
                        logger.info(f'>>>>>>>start training - cell: {i} - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        runner.train()

                        logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        mae, mse, mape = runner.test(load_models=False)
                        mae_list.append(mae)
                        mse_list.append(mse)
                        mape_list.append(mape)

                        if config['train']['do_predict']:
                            logger.info(f'>>>>>>>predicting - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                            runner.predict(load_models=True)

                else:
                    iter_time = 0
                    runner = Runner(iter_time, **config)
                    logger.info(f'>>>>>>>cell: {i} testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.test(load_models=True)
                    torch.cuda.empty_cache()

            f = open('result.txt', 'a')
            f.write(f'mse:{np.mean(mse_list)}, mae:{np.mean(mae_list)}, mape:{np.mean(mape_list)}')
            f.write('\n')
            f.write('\n')
            f.close()

if __name__ == '__main__':
    # generate parser and args
    parser = argparse.ArgumentParser('DP-LET')
    parser.add_argument('--dataset', type=str, default='MILAN', choices=['MILAN', 'TRENTINO'], help='dataset')
    parser.add_argument('--num_cells', type=int, default=100, choices=[100, 200, 300, 400, 500, 600, 700, 800, 900,
                                                                       1000], help='Num of Cells')
    parser.add_argument('--feature', type=str, default='Internet', choices=['SMS-in', 'SMS-out', 'Call-in', 'Call-out',
                                                                             'Internet'], help='feature')
    parser.add_argument('--dim', type=str, default='time_loc', choices=['time', 'time_loc', 'time_feat', 'time_loc_feat'],
                        help='the dimension of data')
    args = parser.parse_args()
    args.config_filename = f'../configs/DP-LET_{args.feature}.yaml'
    logger = get_logger_simple('log', f'DP-LET_{args.dataset}_{args.feature}')
    print(f'Baseline:DP-LET\t-Dataset:{args.feature}\t-num of cells:{args.num_cells}\t-dim:{args.dim}')

    main(args)

