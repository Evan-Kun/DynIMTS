import copy
import datetime
import os
import pathlib
import time
from argparse import ArgumentParser

# Kun
import sys
sys.path.insert(0, '/home/uqkhan/DynGraph')
# sys.path.insert(0, 'C:/Users/EVAN4/Documents/DynGraph')

import numpy as np
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

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def has_graph_support(model_cls):
    return model_cls in [models.GRINCLASSIFIER, models.DGLACLASSIFIER, models.BRITSCLASSIFIER, models.Raindrop]

def get_model_classes(model_str):
    if model_str == 'dglac':
        model, classifier = models.DGLACLASSIFIER, classifiers.GraphClassifier
    elif model_str == 'grinc':
        model, classifier = models.GRINCLASSIFIER, classifiers.GraphClassifier
    elif model_str == 'britsc':
        model, classifier = models.BRITSCLASSIFIER, classifiers.BritsClassifier
    elif model_str == 'raindrop':
        model, classifier = models.Raindrop, classifiers.GraphClassifier
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
    elif dataset_name == 'fm':
        dataset = datasets.fm()
    elif dataset_name == 'NATOPS':
        dataset = datasets.NATOPS()
    elif dataset_name == 'missing_PAM':
        dataset = datasets.MissingPAM(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_fm':
        dataset = datasets.MissingFM(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_NATOPS':
        dataset = datasets.MissingNATOPS(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_HM':
        dataset = datasets.MissingHM(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_SRSCP':
        dataset = datasets.MissingSRSCP(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_HB':
        dataset = datasets.MissingHB(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_SRSCP2':
        dataset = datasets.MissingSRSCP2(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_AWR':
        dataset = datasets.MissingAWR(missing_ratio=args.missing_ratio)
    elif dataset_name == 'missing_BM':
        dataset = datasets.MissingBM(missing_ratio=args.missing_ratio)
    else:
        raise ValueError(f'Dataset {dataset_name} not available in this setting')
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--model-name", type=str, default="dglac")
    parser.add_argument("--dataset-name", type=str, default='air36')
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
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=True)
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

    content = dataset.df.to_numpy()
    n_zeros = np.count_nonzero(content)
    print("missing ratio is: ", args.missing_ratio)
    print("data missing ratio is: ", (content.size - n_zeros) / content.size)

    ##############################################################
    # Create logdir and save configuration
    ##############################################################

    exp_name = f"{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)


    ########################################
    # create result dir and save results
    ########################################
    result_dir = os.path.join(config['logs'], 'classification')
    filename = [args.dataset_name, args.model_name, str(args.batch_size), str(args.lr), str(args.l2_reg), str(args.d_emb),
                str(args.d_model), str(args.d_hidden), str(args.ff_dropout), str(args.missing_ratio),args.adj,
                str(args.adj_threshold),str(args.seed)]
    filename = '_'.join(filename) + '.txt'


    ##############################################################
    # data module
    ##############################################################

    # Instantiate dataset
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
    elif args.adj == 'he':
        scale = np.sqrt(2.0 / adj.shape[0])
        adj = np.random.randn(adj.shape[0], adj.shape[0])*scale
    # np.fill_diagonal(adj, 0.)
    # adj_sum = np.sum(adj, axis=1)
    # adj = adj / adj_sum

    # force adj with no self loop

    #################################################################
    # predictor
    #################################################################

    # model's input
    # additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)

    # BRITS
    if args.model_name == 'britsc':
        additional_model_hparams = dict(adj=adj, d_in=args.d_dim, n_nodes=dm.n_nodes)
    else:
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

    if has_graph_support(model_cls):
        y_train = tuple(np.array([sample[0]['y'][0, -1].numpy()[0] for sample in dm.trainset]))
    else:
        y_train = tuple(np.array([sample[0]['y'][0, -1].numpy() for sample in dm.trainset]))
    y_classes = tuple(np.unique(y_train))
    class_weight = torch.tensor(compute_class_weight('balanced', classes=y_classes, y=y_train), dtype=torch.float)

    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5816, 3.5640]))
    # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.02, 0.98]))
    # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.52187, 11.93204]))
    # loss_fn = torch.nn.BCELoss(weight=class_weight)
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([0.58, 3.51]))
    #
    if args.n_class > 2:

        metrics =  {'accuracy': Accuracy(compute_on_step=False, num_classes=args.n_class),
                   'precision': Precision(compute_on_step=False, num_classes=args.n_class),
                   'recall': Recall(compute_on_step=False, num_classes=args.n_class),
                   'F1': F1(compute_on_step=False, num_classes=args.n_class),
                   # 'AUC': AUC(compute_on_step=False, reorder=True, ),
                   'AUROC': AUROC(compute_on_step=False, num_classes=args.n_class),
                   'AP': AveragePrecision(compute_on_step=False, pos_label=1, num_classes=args.n_class)
                    }
    else:
        metrics =  {'accuracy': Accuracy(compute_on_step=False, multiclass=True),
                   'precision': Precision(compute_on_step=False, multiclass=True),
                   'recall': Recall(compute_on_step=False, num_classes=args.n_class, multiclass=True),
                   'F1': F1(compute_on_step=False, num_classes=args.n_class, multiclass=True),
                    'AUC': AUC(compute_on_step=False, reorder=True),
                   'AUROC': AUROC(compute_on_step=False,  num_classes=1),
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
                                            #  'eta_min': 0.0001,
                                            'eta_min': 0.000001,
                                             'T_max': args.epochs,
                                             'verbose': True
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



    # classifier.class_weight = class_weight

    # Calculate the class weight
    # trainset = dm.trainset
    # for sample in trainset:
    #     sample_data = sample[0]['y']
    #     sample_data_col = sample_data[0, :]
    #     sample_data_0 = sample_data[0, :][0]
    #     sample_data_1 = sample_data[0, :][-1]
    #     if sample[0]['y'][0, 0, -1] != 0:
    #         print(sample[0]['y'])
    #     y = sample[0]['y'][0, 0, -1]


    ######################################################################
    # training
    ######################################################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_AUROC', patience=args.patience, mode='max')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_AUROC', mode='max')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         # track_grad_norm=2,
                         deterministic=True)
    
    # Measure start time
    start_time = time.time()

    trainer.fit(classifier, datamodule=dm)

    # Measure end time
    end_time = time.time()
    # Calculate and print the running time
    running_time = end_time - start_time
    print(f"Training completed in {running_time:.2f} seconds")

    ####################################################################
    #  Testing
    ####################################################################

    classifier.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                          lambda storage, loc: storage)['state_dict'])
    classifier.freeze()

    # Measure start time for test
    start_time_test = time.time()

    trainer.test()
    classifier.eval()

    # Measure end time for evaluation
    end_time_test = time.time()

    # Calculate and print the evaluation time
    test_time = end_time_test - start_time_test
    print(f"Evaluation completed in {test_time:.2f} seconds")   
    
    if torch.cuda.is_available(): classifier.cuda()

    with torch.no_grad():
        y_true, y_predict = classifier.predict_loader(dm.test_dataloader())

    # y_predict = y_predict.detach().cpu().numpy.reshape(y_predict.shape[:3])


    if args.n_class == 2:
        y_pred = torch.softmax(y_predict, dim=1)
        y_prob = torch.softmax(y_predict, dim=1)[:, 1]
        y_prob = y_prob.detach().cpu()
        y_prob_ob = y_prob.numpy().astype(int)
    else:
        y_pred = torch.softmax(y_predict, dim=1)

    y_true = y_true.detach().cpu()
    y_pred= y_pred.detach().cpu()

    # observation
    y_true_ob = y_true.numpy()
    y_predict_ob = y_pred.numpy()
    y_predict_label_ob = np.argmax(y_predict_ob, axis=1)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import confusion_matrix

    if args.n_class == 2:
        accuracy_score = accuracy_score(y_true_ob, y_predict_label_ob)
        precision_score = precision_score(y_true_ob, y_predict_label_ob)
        f1_score = f1_score(y_true_ob, y_predict_label_ob)
        recall_score = recall_score(y_true_ob, y_predict_label_ob)
        roc_auc_score = roc_auc_score(y_true_ob, y_prob)
        average_precision_score = average_precision_score(y_true_ob, y_prob)
        cm = confusion_matrix(y_true_ob, y_predict_label_ob)
        # accuracy_score = accuracy_score(y_true_ob, y_prob_ob)
        # precision_score = precision_score(y_true_ob, y_prob_ob)
        # f1_score = f1_score(y_true_ob, y_prob_ob)
        # recall_score = recall_score(y_true_ob, y_prob_ob)
        # accuracy_score = accuracy_score(y_true_ob, y_prob_ob)
        # precision_score = precision_score(y_true_ob, y_prob_ob)
        # f1_score = f1_score(y_true_ob, y_prob_ob)
        # recall_score = recall_score(y_true_ob, y_prob_ob)
        # roc_auc_score = roc_auc_score(y_true_ob, y_prob_ob)
        # average_precision_score = average_precision_score(y_true_ob, y_prob_ob)

    else:
        accuracy_score = accuracy_score(y_true_ob, y_predict_label_ob)
        precision_score = precision_score(y_true_ob, y_predict_label_ob, average='macro')
        f1_score = f1_score(y_true_ob, y_predict_label_ob, average='macro')
        recall_score = recall_score(y_true_ob, y_predict_label_ob, average='macro')
        cm = confusion_matrix(y_true_ob, y_predict_label_ob)

    print("Sklearn metircs: ")
    print("accuracy: ", accuracy_score)
    print("precision_score: ", precision_score)
    print("recall_score: ", recall_score)
    print("f1_score: ", f1_score)
    print("roc_auc_score: ", roc_auc_score)
    print("average_precision_score: ", average_precision_score)
    print("confusion matrix: \n", cm)

    with open(os.path.join(result_dir, filename), 'w') as r:
    #     for metric_name, metric_fn in metrics.items():
    #         metric_fn.update(y_predict, y_true)
    #         error = metric_fn.compute().numpy()
    #         print(f' {metric_name}: {error:.4f}')
        r.write(f'"accuracy: ": {accuracy_score:.4f} \n')
        r.write(f'"precision_score: ": {precision_score:.4f} \n')
        r.write(f'"recall_score: ": {recall_score:.4f} \n')
        r.write(f'"f1_score: ": {f1_score:.4f} \n')
        # r.write(f'"cm: ": {cm:.4f} \n')
        if args.n_class == 2:
            r.write(f'"roc_auc_score: ": {roc_auc_score:.4f} \n')
            r.write(f'"average_precision_score: ": {average_precision_score:.4f} \n')
        r.write(f'"Training: ": {running_time:.2f} \n')
        r.write(f'"Testing: ": {test_time:.2f} \n')

    # Test classification in whole
    # eval_mask = dataset.eval_mask[dm.test_slice]
    # df_true = dataset.df.iloc[dm.test_slice]
    # metrics = {
    #     'acc': accuracy_score,
    #     'f1': f1_score,
    #     'precision': precision_score,
    #     'recall': recall_score
    # }

    # Aggregate predictions in dataframes
    # index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    # aggr_methods = ensure_list(args.aggregate_by)
    # df_hats = prediction_dataframe(y_predict, index, dataset.df.columns, aggregate_by=aggr_methods)
    # df_hats = dict(zip(aggr_methods, df_hats))

    # with open(os.path.join(result_dir, filename), 'w') as r:
    #     for aggr_by, df_hat in df_hats.items():
    #         # Compute error
    #         print(f'- AGGREGATE BY {aggr_by.upper()}')
    #         for metric_name, metric_fn in metrics.items():
    #             error = metric_fn(df_hat.values, df_true.values).item()
    #             print(f' {metric_name}: {error:.4f}')
    #             r.write(f'{metric_name}: {error:.4f} \n')
    # with open(os.path.join(result_dir, filename), 'w') as r:
    #     for metric_name, metric_fn in metrics.items():
    #         metric_fn.update(y_predict, y_true)
    #         error = metric_fn.compute().numpy()
    #         print(f' {metric_name}: {error:.4f}')
    #         r.write(f'{metric_name}: {error:.4f} \n')

    return y_true, y_predict


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)