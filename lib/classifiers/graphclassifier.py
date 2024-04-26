import torch

from . import Classifier
from ..nn.models import MPGRUNet, GRINet, DGLANet, DGRINet, DGLACLASSIFIER

from torchmetrics.functional.classification import accuracy
from torchmetrics.functional import f1
from torchmetrics.functional.classification import recall
from torchmetrics.functional.classification import precision
from torchmetrics.functional.classification import auc
from torchmetrics.functional.classification import roc
from torchmetrics.functional.classification import auroc

import numpy as np

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


class GraphClassifier(Classifier):

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 warm_up=0,
                 pred_loss_weight=1,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 n_class=2):
        super(GraphClassifier, self).__init__(model_class=model_class,
                                              model_kwargs=model_kwargs,
                                              optim_class=optim_class,
                                              optim_kwargs=optim_kwargs,
                                              loss_fn=loss_fn,
                                              metrics=metrics,
                                              scheduler_class=scheduler_class,
                                              scheduler_kwargs=scheduler_kwargs,
                                              )

        self.tradeoff = pred_loss_weight
        if model_class is MPGRUNet:
            self.trimming = (warm_up, 0)
        elif model_class in [GRINet, DGLANet, DGRINet, DGLACLASSIFIER]:
            self.trimming = (warm_up, warm_up)
        self.n_class = n_class

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq


    def training_step(self, batch, batch_idx):
        # unpack_batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # compute masks
        # mask = batch_data['mask'].clone().detach()
        # eval_mask = batch_data.pop('eval_mask', None)
        # eval_mask = mask | eval_mask

        y = batch_data.pop('y')
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)

        # compute predictions and compute loss
        predictions, states, y_predict = self.predict_batch(batch, preprocess=False)

        # trim to
        # predictions = self.trim_seq(*predictions)

        target = y.type(torch.LongTensor).cuda()
        # classify_mask = torch.ones(y.shape[0], y.shape[1])
        loss = self.loss_fn(y_predict, target)

        # logging
        # y_probability = y_predict[:, 1]
        if self.n_class == 2:
            y_probability = torch.sigmoid(y_predict)[:, 1]
            self.train_metrics.update(y_probability.detach(), target)
        else:
            y_probability = torch.softmax(y_predict, dim=1)
            self.train_metrics.update(one_hot(y_probability.detach()), target)

        probability_loss = torch.squeeze(torch.sum(y_probability - target * y_probability)/y_probability.shape[0])
        loss += probability_loss
        # self.log_dict(grad_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        # mask = batch_data.get('mask')
        # eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)
        # Compute predictions and loss
        predictions, y_predict = self.predict_batch(batch, preprocess=False)

        # trim to
        # predictions = self.trim_seq(*predictions)

        target = y.type(torch.LongTensor).cuda()
        # y_predict = y_predict[:, 1]
        # classify_mask = torch.ones(y.shape[0])
        # val_loss = self.loss_fn(y_predict, target, classify_mask, task='binary')



        # test_bloss = binary_accuracy(y_predict, target)
        # test_bloss = BinaryAccuracy(y_predict, target)
        # test_loss = accuracy(y_predict, target, num_classes=None)


        # loss = torch.nn.CrossEntropyLoss()
        val_loss = self.loss_fn(y_predict, target)
        # val_loss = self.loss_fn(y_predict, target, classify_mask)
        # Logging
        # y_probability = y_predict[:, 1]
        if self.n_class == 2:
            y_probability = torch.sigmoid(y_predict)[:, 1]
            self.val_metrics.update(y_probability.detach(), target)
        else:
            y_probability = torch.softmax(y_predict, dim=1)
            self.val_metrics.update(one_hot(y_probability.detach()), target)
        # y_predict = torch.argmax(y_predict, dim=1)
        # acc = accuracy(y_probability, target)
        # f1_score = f1(y_probability, target)
        # auc_score = auc(y_probability, target, reorder=True)
        # roc_score = roc(y_probability, target)
        # auroc_score = auroc(y_probability, target, pos_label=1)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        # eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)
        # classify_mask = torch.ones(y.shape[0], y.shape[1])

        # Compute outputs
        predictions, y_predict = self.predict_batch(batch, preprocess=False)

        target = y.type(torch.LongTensor).cuda()
        test_loss = self.loss_fn(y_predict, target)

        # Logging
        # y_probability = y_predict[:, 1]
        if self.n_class == 2:
            y_probability = torch.sigmoid(y_predict)[:, 1]
            self.test_metrics.update(y_probability.detach(), target)
        else:
            y_probability = torch.softmax(y_predict, dim=1)
            self.test_metrics.update(one_hot(y_probability.detach()), target)

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss

