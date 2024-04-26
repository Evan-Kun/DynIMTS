import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

# Kun
import sys
# sys.path.insert(0, '/home/s4516787/grin')
sys.path.insert(0, '/home/uqkhan/grin')

import numpy as np
np.set_printoptions(threshold=np.inf)


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool



def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.DGRINet, models.MPGRUNet, models.BiMPGRUNet, models.DGLANet]


def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    elif model_str == 'dgrin':
        model, filler = models.DGRINet, fillers.GraphFiller
    elif model_str == 'dgla':
        model, filler = models.DGLANet, fillers.GraphFiller
    elif model_str == 'mpgru':
        model, filler = models.MPGRUNet, fillers.GraphFiller
    elif model_str == 'bimpgru':
        model, filler = models.BiMPGRUNet, fillers.GraphFiller
    elif model_str == 'var':
        model, filler = models.VARImputer, fillers.Filler
    elif model_str == 'gain':
        model, filler = models.RGAINNet, fillers.RGAINFiller
    elif model_str == 'birnn':
        model, filler = models.BiRNNImputer, fillers.MultiImputationFiller
    elif model_str == 'rnn':
        model, filler = models.RNNImputer, fillers.Filler
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay_block':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la_block':
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.75)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif dataset_name == 'synthetic':
        # dataset = datasets.SyntheticDataset(p_block=0, p_point=0.2, window=32)
        # dataset = datasets.SyntheticData(missing_ratio=0.8)
        dataset = datasets.MissingSynthetic(missing_ratio=args.missing_ratio)
    elif dataset_name == 'exchange_rate':
        dataset = datasets.MissingExchangeRate(missing_ratio=args.missing_ratio)
    elif dataset_name == 'solar':
        dataset = datasets.MissingSolar(missing_ratio=args.missing_ratio)
    elif dataset_name == 'electricity':
        dataset = datasets.MissingElectricity(missing_ratio=args.missing_ratio)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='brits')
    parser.add_argument("--dataset-name", type=str, default='air36')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)
    parser.add_argument('--adj', type=str, default='physical')
    parser.add_argument('--missing-ratio', type=float, default=0.2)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)


    ########################################
    # create result dir and save results
    ########################################
    result_dir = os.path.join(config['logs'], 'results')
    filename = [args.dataset_name, args.model_name, str(args.d_hidden), args.adj, str(args.missing_ratio), str(args.seed)]
    filename = '_'.join(filename) + '.txt'



    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    if args.adj == 'static':
        adj[:,:] = 1
    elif args.adj == 'ground-truth':
        adj = np.array([[0, 0.8, 0, 0, 0, 0.5], [0.8, 0, 0.2, 0, 0, 0.46],
                     [0, 0.2, 0, 0.8, 0, 0], [0, 0, 0.9, 0, 0.1, 0],
                     [0, 0, 0, 0.1, 0, 0.9],[0.46, 0.46, 0, 0, 0.9, 0]], np.float)
        adj = adj.T
    elif args.adj == 'correlation':
        adj[:,:] = dataset.get_correlation()
    elif args.adj == 'zero':
        adj[:, :] = 0
    elif args.adj == '25':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.25)
        adj[:partition,:partition] = 1
    elif args.adj == '50':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.50)
        adj[:partition,:partition] = 1
    elif args.adj == '75':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.75)
        adj[:partition,:partition] = 1
    elif args.adj == '-25':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.25)
        adj[-partition:,-partition:] = 1
    elif args.adj == '-50':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.50)
        adj[-partition:,-partition:] = 1
    elif args.adj == '-75':
        adj[:, :] = 0
        partition = round(adj.shape[0] * 0.75)
        adj[-partition:,-partition:] = 1
    elif args.adj == 'random':
        adj[:, :] = 0
        adj_random = np.random.uniform(0, 1, (adj.shape[0], adj.shape[1]))
        adj = adj_random

    # force adj with no self loop
    np.fill_diagonal(adj, 0.)
    # adj = np.tile(adj, [32, 1, 1])
    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg,
                                                   },
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         deterministic=True)

    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test()
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))

    with open(os.path.join(result_dir, filename), 'w') as r:
        for aggr_by, df_hat in df_hats.items():
            # Compute error
            print(f'- AGGREGATE BY {aggr_by.upper()}')
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
                r.write(f'{metric_name}: {error:.4f} \n')

    # print(y_hat)
    # print(y_true)
    # Assuming df_hat and df_true are your DataFrames
    import pandas as pd

    eval_mask = eval_mask.astype(bool)
    # Mask to identify where the actual imputation took place
    imputed_values_hat = df_hat.where(eval_mask)
    ground_truth_values = df_true.where(eval_mask)

    # Calculate differences between imputed and true values
    differences = imputed_values_hat - ground_truth_values
    error_stats = differences.describe()
    print("Error Statistics:")
    print(error_stats)
    # Define thresholds based on percentiles of ground-truth values
    threshold_small = df_true.quantile(0.25)
    threshold_large = df_true.quantile(0.75)

    # Create masks for large and small values in the ground-truth data
    mask_large_values = df_true > threshold_large
    mask_small_values = df_true < threshold_small

    # Analyze differences where original values are large
    large_value_differences = differences.where(mask_large_values & eval_mask)
    # print("Differences for large original values:")
    # print(large_value_differences.describe())
    # print(large_value_differences)

    # Analyze differences where original values are small
    small_value_differences = differences.where(mask_small_values & eval_mask)
    # print("Differences for small original values:")
    # print(small_value_differences.describe())
    # print(small_value_differences)

    # Drop rows where all values are NaN in small_value_differences
    small_value_differences_cleaned = small_value_differences.dropna(how='all')

    # Drop rows where all values are NaN in large_value_differences
    large_value_differences_cleaned = large_value_differences.dropna(how='all')

    # Now you can print or analyze these cleaned DataFrames
    print("Cleaned Differences for Large Original Values:")
    print(large_value_differences_cleaned.describe())
    print(large_value_differences_cleaned)

    print("Cleaned Differences for Small Original Values:")
    print(small_value_differences_cleaned.describe())
    print(small_value_differences_cleaned)

    # Display only rows with differences, with more context
    # if not differences.empty:
    #     # Optionally, add some context like index or specific columns to better understand the differences
    #     context = df_hat.join(df_true, lsuffix='_hat', rsuffix='_true', how='inner')
    #     detailed_view = context[context.index.isin(differences.index)]
    #     print(detailed_view)
    # else:
    #     print("No differences found.")
    
    return y_true, y_hat, mask


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
