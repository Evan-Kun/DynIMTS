import inspect
from copy import deepcopy

# import pylab as p
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics import MetricCollection
from pytorch_lightning.utilities import move_data_to_device
from ..nn.utils.metric_base import MaskedMetric

from ..nn.models.brits_classifier import BRITSCLASSIFIER

from .. import epsilon
from .. utils.utils import ensure_list

class Classifier(pl.LightningModule):
    def __init__(self,
                model_class,
                model_kwargs,
                optim_class,
                optim_kwargs,
                loss_fn,
                metrics=None,
                scheduler_class=None,
                scheduler_kwargs=None):

        """
        PL module to implement hole fillers.

        :param model_class: Class of pytorch nn.Module implementing the classifer.
        :param model_kwargs: Model's keyword arguments.
        :param optim_class: Optimizer class.
        :param optim_kwargs: Optimizer's keyword arguments.
        :param loss_fn: Loss function used for training.
        :param metrics: Dictionary of type {'metric1_name':metric1_fn, 'metric2_name':metric2_fn ...}.
        :param scheduler_class: Scheduler class.
        :param scheduler_kwargs: Scheduler's keyword arguments.
        """

        super(Classifier, self).__init__()
        self.save_hyperparameters(model_kwargs)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs =  scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        if metrics is None:
            metrics = dict()

        self._set_metrics(metrics)
        # instantiate model
        self.model = self.model_cls(**self.model_kwargs)

    def reset_model(self):
        self.model = self.model_cls(**self.model_kwargs)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @auto_move_data
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def _check_metric(metric, on_step=False):
        # if not isinstance(metric, MaskedMetric):
        #     if 'reduction' in inspect.getfullargspec(metric).args:
        #         metric_kwargs = {'reduction': 'none'}
        #     else:
        #         metric_kwargs = dict()
        #     return MaskedMetric(metric, compute_on_step=on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            {f'train_{k}': self._check_metric(m, on_step=True) for k, m in metrics.items()})
        self.val_metrics = MetricCollection({f'val_{k}': self._check_metric(m) for k, m in metrics.items()})
        self.test_metrics = MetricCollection({f'test_{k}': self._check_metric(m) for k, m in metrics.items()})

    def _preprocess(self, data, batch_preprocessing):
        """
        Perform preprocessing of a given input

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: preprocessed data
        """
        if isinstance(data, (list, tuple)):
            return [self._preprocess(d, batch_preprocessing) for d in data]
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)
        return (data - trend - bias) / (scale + epsilon)

    # def _postprocess(self, data, batch_preprocessing):
    #     """
    #     Perform preprocessing(inverse transform) of a given input
    #
    #     :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
    #     :param batch_preprocessing: dictionary containing preprocessing data
    #     :return: inverse transformed data
    #     """
    #     if isinstance(data, (list, tuple)):
    #         return [self._postprocess(d, batch_preprocessing) for d in data]
    #     trend = batch_preprocessing.get('trend', 0.)
    #     bias = batch_preprocessing.get('bias', 0.)
    #     scale = batch_preprocessing.get('scale', 1.)
    #     return data * (scale + epsilon) + bias + trend

    def predict_batch(self, batch, preprocess=False, return_target=False):
        """
        This method takes an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Predictions should have a shape [batch, node, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...}
                                                                preprocessing:
                                                                {'bias':..., 'scale':..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that the inputs are by default preprocessed before creating the batch)
        # :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the transformed signal)
        :param return_target: whether to return the prediction target_y_true and the prediction mask
        :return: (y_ture), y_predict, (mask)
        """

        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop('x')
            x = self._preprocess(x, batch_preprocessing)
            y_predict = self.forward(x, **batch_data)
        else:
            y_predict = self.forward(**batch_data)

        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, y_predict, mask
        return y_predict

    def predict_loader(self, loader, preprocess=False):
        """
        Makes predictions for an input dataloader. Returns both the predictions and predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_ture, y_predict
        """

        targets, predicts, repres = [], [] ,[]
        for batch in loader:
            batch = move_data_to_device(batch, self.device)
            batch_data, batch_preprocessing = self._unpack_batch(batch)
            # Extract mask and target
            # eval_mask = batch_data.pop('eval_mask', None)
            y = batch_data.pop('y')

            # Others
            y = y[:, 0, -1, :]
            y = torch.squeeze(y)

            # predictions, y_predict, y_repre = self.predict_batch(batch, preprocess=preprocess)
            # if isinstance(y_predict, (list, tuple)):
            #     y_predict = y_predict[0]
            # # y_repre = torch.softmax(y_repre, dim=1)
            # predicts.append(y_predict)
            # repres.append(y_repre)
            # masks.append(eval_mask)

            # BRITS
            if isinstance(self.model, BRITSCLASSIFIER):
                # y = y[:, 0, -1]
                # y = torch.squeeze(y)
                predictions, y_repre = self.predict_batch(batch, preprocess=preprocess)
                repres.append(y_repre)
            else:
                predictions, y_predict, y_repre = self.predict_batch(batch, preprocess=preprocess)
                if isinstance(y_predict, (list, tuple)):
                    y_predict = y_predict[0]
                # # y_repre = torch.softmax(y_repre, dim=1)
                predicts.append(y_predict)
                repres.append(y_repre)

            target = y.type(torch.LongTensor).cuda()
            targets.append(target)
        print("Predicted loader")
        y = torch.cat(targets, 0)
        y_pre = torch.cat(repres, 0)
        return y, y_pre

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries

        :param batch: batch
        :return: batch_data, batch_preprocessing
        """
        if isinstance(batch, (tuple, list)) and (len(batch) == 2):
            batch_data, batch_preprocessing = batch
        else:
            batch_data = batch
            batch_preprocessing = dict()
        return batch_data, batch_preprocessing

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data['mask'].clone.detach()
        # batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob)
        eval_mask = batch_data.pop('eval_mask')
        eval_mask = (mask | eval_mask) - batch_data['mask']

        y = batch_data.pop('y')

        # compute predictions and compute loss
        predictions = self.predict_batch(batch, preprocess=False)



        # logging
        self.train_metrics.update(predictions, y, classify_mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # compute predictions and target
        predictions = self.predict_batch(batch, preprocess=False)

        target = y
        classify_mask = torch.ones(y.shape[0], y.shape[1])
        val_loss = self.loss_fn(predictions, target, classify_mask)

        # logging
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # unpack_batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract batch
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        classify_mask = torch.ones(y.shape[0], y.shape[1])
        # compute outputs and target
        predictions = self.predict_batch(batch, preprocess=False)
        test_loss = self.loss_fn(predictions, y, classify_mask)

        # logging
        self.test_metrics.update(predictions.detach(), y, classify_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return test_loss

    def on_train_epoch_start(self) -> None:
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['metric'] = metric
        return cfg

