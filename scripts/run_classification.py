import numpy as np
import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

# Kun
import sys
# sys.path.insert(0, '/home/s4516787/grin')
sys.path.insert(0, '/home/uqkhan/grin')

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib import classifiers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.nn.utils.metrics import MaskedAccuracy, MaskedPrecision, MaskedRecall, MaskedF1, MaskedAUC, MaskedROC, MaskedAUROC
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from torchmetrics import Accuracy, F1, Recall, Precision, AUC, AUROC, AveragePrecision



def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.DGRINet, models.MPGRUNet, models.BiMPGRUNet, models.DGLACLASSIFIER]


def get_model_classes(model_str):
    if model_str == 'dglac':
        model, classifier = models.DGLACLASSIFIER, classifiers.GraphClassifier
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, classifier


def get_dataset(dataset_name):
    if dataset_name == 'P19':
        dataset = datasets.P19()
    elif dataset_name == 'P12':
        dataset = datasets.P12()
    elif dataset_name == 'PAM':
        dataset = datasets.PAM()
    else:
        raise ValueError(
            f'Dataset {dataset_name} not available in this setting')
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--model-name", type=str, default="dglac")
    parser.add_argument("--dataset-name", type=str, default='P12')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)
    parser.add_argument("--aggregate-by", type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='cross_entropy')
    parser.add_argument('--use-lr-schedule', type=str_to_bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    parser.add_argument('--adj', type=str, default='physical')
    parser.add_argument('--missing-ratio', type=float, default=0.2)

    parser.add_argument('--n-class', type=int, default=2)

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

    model_cls, classifier_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)


    print(dataset)
    ##############################################################
    # Create logdir and save configuration
    ##############################################################

    exp_name = f"{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(
        config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(
            args), fp, indent=4, sort_keys=True)

    ########################################
    # create result dir and save results
    ########################################
    result_dir = os.path.join(config['logs'], 'classification')
    filename = [args.dataset_name, args.model_name, str(
        args.d_hidden), args.adj, str(args.missing_ratio), str(args.seed)]
    filename = '_'.join(filename) + '.txt'

    ##############################################################
    # data module
    ##############################################################

    # Instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(
        model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(
        args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(
        torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(
        args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    if args.adj == 'static':
        adj[:, :] = 1
    elif args.adj == 'ground-truth':
        adj = np.array([[0, 0.8, 0, 0, 0, 0.5], [0.8, 0, 0.2, 0, 0, 0.46],
                        [0, 0.2, 0, 0.8, 0, 0], [0, 0, 0.9, 0, 0.1, 0],
                        [0, 0, 0, 0.1, 0, 0.9], [0.46, 0.46, 0, 0, 0.9, 0]], np.float)
        adj = adj.T
    elif args.adj == 'correlation':
        adj[:, :] = dataset.get_correlation()
    elif args.adj == 'zero':
        adj[:, :] = 0
    np.fill_diagonal(adj, 0.)
    # force adj with no self loop

    #################################################################
    # predictor
    #################################################################

    # model's input
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    # loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
    #                        compute_on_step=True,
    #                        metric_kwargs={'reduction':'none'})
    #
    #
    # metrics = {'accuracy': MaskedAccuracy(compute_on_step=False),
    #            'precision': MaskedPrecision(compute_on_step=False),
    #            'recall': MaskedRecall(compute_on_step=False),
    #            'F1': MaskedF1(compute_on_step=False),
    #            'AUC': MaskedAUC(compute_on_step=False),
    #            'ROC': MaskedROC(compute_on_step=False),
    #            'AUROC': MaskedAUROC(compute_on_step=False),}

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.n_class > 2:

        metrics = {'accuracy': Accuracy(compute_on_step=False, num_classes=args.n_class),
                   'precision': Precision(compute_on_step=False, num_classes=args.n_class),
                   'recall': Recall(compute_on_step=False, num_classes=args.n_class),
                   'F1': F1(compute_on_step=False, num_classes=args.n_class),
                   'AUC': AUC(compute_on_step=False, reorder=True),
                   'AUROC': AUROC(compute_on_step=False, num_classes=args.n_class),
                   'AP': AveragePrecision(compute_on_step=False, pos_label=1, num_classes=args.n_class)
                   }
    else:
        metrics = {'accuracy': Accuracy(compute_on_step=False),
                   'precision': Precision(compute_on_step=False),
                   'recall': Recall(compute_on_step=False),
                   'F1': F1(compute_on_step=False),
                   'AUC': AUC(compute_on_step=False, reorder=True),
                   'AUROC': AUROC(compute_on_step=False),
                   'AP': AveragePrecision(compute_on_step=False, pos_label=1)
                   }
    # classifier inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_classifier_hparams = dict(model_class=model_cls,
                                         model_kwargs=model_kwargs,
                                         optim_class=torch.optim.Adam,
                                         optim_kwargs={'lr': args.lr,
                                                       'weight_decay': args.l2_reg},
                                         loss_fn=loss_fn,
                                         metrics=metrics,
                                         scheduler_class=scheduler_class,
                                         scheduler_kwargs={
                                             'eta_min': 0.0001,
                                             'T_max': args.epochs
                                         },
                                         alpha=args.alpha,
                                         hint_rates=args.hint_rate,
                                         g_train_freq=args.g_train_freq,
                                         d_train_freq=args.d_train_freq,
                                         n_class=args.n_class)
    classifier_kwargs = parser_utils.filter_args(args={**vars(args), **additional_classifier_hparams},
                                                 target_cls=classifier_cls,
                                                 return_dict=True)
    classifier = classifier_cls(**classifier_kwargs)

    ######################################################################
    # training
    ######################################################################

    # callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_AUROC', patience=args.patience, mode='max')
    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir, save_top_k=1, monitor='val_AUROC', mode='max')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         track_grad_norm=2,
                         deterministic=True)

    trainer.fit(classifier, datamodule=dm)

    ####################################################################
    #  Testing
    ####################################################################

    classifier.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                          lambda storage, loc: storage)['state_dict'])
    classifier.freeze()
    trainer.test()
    classifier.eval()

    if torch.cuda.is_available():
        classifier.cuda()

    with torch.no_grad():
        y_true, y_predict, mask = classifier.predict_loader(
            dm.test_dataloader(), return_mask=True)

    y_predict = y_predict.detach().cpu().numpy.reshape(y_predict.shape[:3])

    # Test classification in whole
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        'acc': numpy_metrics.masked_accuracy,
        'f1': numpy_metrics.masked_f1,
        'auc': numpy_metrics.masked_auc,
        'roc': numpy_metrics.masked_roc,
        'auroc': numpy_metrics.masked_auroc,
    }

    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(
        dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(
        y_predict, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))

    with open(os.path.join(result_dir, filename), 'w') as r:
        for aggr_by, df_hat in df_hats.items():
            # Compute error
            print(f'- AGGREGATE BY {aggr_by.upper()}')
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(
                    df_hat.values, df_true.values, eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
                r.write(f'{metric_name}: {error:.4f} \n')

    return y_true, y_predict, mask


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
