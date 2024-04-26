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
import os

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

        # oversampling
        # idx_1 = torch.nonzero(y).squeeze().float()
        # bal_idx_1 = torch.multinomial(idx_1, int(len(y)/2), replacement=True)
        # idx_0 = (y == 0.0).nonzero().squeeze().float()
        # bal_idx_0 = torch.multinomial(idx_0, int(len(y)/2), replacement=True)
        #
        # bal_n = batch[0]['x'][bal_idx_0,:,:,:]
        # bal_p = batch[0]['x'][bal_idx_1,:,:,:]
        # mask_n = batch[0]['mask'][bal_idx_0,:,:,:]
        # mask_p = batch[0]['mask'][bal_idx_1,:,:,:]
        # batch[0]['x'] = torch.vstack((bal_n, bal_p))
        # batch[0]['mask'] = torch.vstack((mask_n, mask_p))
        # y[:int(len(y)/2)] = 0
        # y[int(len(y)/2):] = 1

        # compute predictions and compute loss
        if isinstance(self.model, DGLACLASSIFIER):
            predictions, states, y_predict, y_repre, learned_adjs = self.predict_batch(batch, preprocess=False)
            if self.current_epoch % 2 == 0 and batch_idx == 0:
                filename = os.path.join(
                    '../logs/P19/learning/', f"PAM_epoch_{self.current_epoch}_batch_{batch_idx}.npy")
                # filename = os.path.join('../logs/P19/learning/', f"P19_epoch_{self.current_epoch}_batch_{batch_idx}.npy")
                # print(learned_adjs)
                adj_arrays = [tensor.detach().cpu().numpy() for tensor in learned_adjs]
                np.save(filename, adj_arrays)
        else:
            predictions, states, y_predict, y_repre = self.predict_batch(batch, preprocess=False)
        # trim to
        # predictions = self.trim_seq(*predictions)
        target = y.type(torch.LongTensor).cuda()
        loss = self.loss_fn(y_predict.float(), target)
        loss += self.loss_fn(y_repre.float(), target)

        # loss = self.loss_fn(y_repre.float(), target)

        # Focal loss
        # loss = self.loss_fn(y_repre, target)
        # pt = torch.exp(-loss)
        # focal_loss = 1 * (1 - pt)** 2 * loss
        # loss = focal_loss.mean()

        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.train_metrics.update(y_predicts.detach(), target)


        # target = y.cuda()
        # target = torch.nn.functional.one_hot(target, num_classes=self.n_class)
        # classify_mask = torch.ones(y.shape[0], y.shape[1])
        # BCELoss
        # m = torch.nn.Sigmoid()
        # loss = self.loss_fn(m(y_predict).squeeze(), target)

        # Cross entropy loss
        # loss =0.1 * self.loss_fn(y_predict, target)
        # loss += self.loss_fn(y_repre, target)



        # grin loss
        # fwd_predict, bwd_predict = self.predict_batch(batch, preprocess=False)
        # loss = self.loss_fn(fwd_predict, target)
        # loss += self.loss_fn(bwd_predict, target)

        # logging
        # y_predict = fwd_predict
        # y_probability = y_predict[:, 1]
        # y_probability = torch.sigmoid(y_predict)


        # if self.n_class == 2:
        #     # y_probability = torch.sigmoid(y_predict)[:, 1]
        #     # y_probability = torch.argmax(torch.sigmoid(y_predict), dim=1)
        #     # self.train_metrics.update(y_probability.detach(), target)
        #     y_probability = torch.sigmoid(y_predict)
        #     y_predicts = y_probability >= 0.5
        #     for metric in self.train_metrics:
        #         metric.update(y_probability.detach(), target)
        #         # metric.update(y_predicts.detach(), target)
        # else:
        #     y_probability = torch.softmax(y_predict, dim=1)
        #     self.train_metrics.update(one_hot(y_probability.detach()), target)
        # #
        # probability_loss = torch.squeeze(torch.sum(y_probability - target * y_probability)/y_probability.shape[0])
        # loss += probability_loss
        # self.log_dict(grad_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        # self.log.experiment.add_images('learned_adj', learned_adjs)
        # self.logger.experiment.add_images('learned_adj', learned_adjs)
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
        predictions, y_predict, y_repre = self.predict_batch(batch, preprocess=False)

        # trim to
        # predictions = self.trim_seq(*predictions)

        target = y.type(torch.LongTensor).cuda()
        # target = y.cuda()
        # target = torch.nn.functional.one_hot(target, num_classes=self.n_class)
        # y_predict = y_predict[:, 1]


        val_loss = self.loss_fn(y_repre.float(), target)

        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.val_metrics.update(y_predicts.detach(), target)
        # classify_mask = torch.ones(y.shape[0])
        # val_loss = self.loss_fn(y_predict, target, classify_mask, task='binary')

        # BCELoss
        # m = torch.nn.Sigmoid()
        # val_loss = self.loss_fn(m(y_predict).squeeze(), target)

        # loss = torch.nn.CrossEntropyLoss()
        # val_loss = self.loss_fn(y_repre, target)
        # val_loss = self.loss_fn(y_predict, target, classify_mask)

        # grin loss
        # fwd_predict, bwd_predict = self.predict_batch(batch, preprocess=False)
        # val_loss = self.loss_fn(fwd_predict, target)
        # val_loss += self.loss_fn(bwd_predict, target)

        # Logging
        # y_predict = fwd_predict
        # y_probability = y_predict[:, 1]


        # if self.n_class == 2:
        #     # y_probability = torch.argmax(torch.sigmoid(y_predict), dim=1)
        #     # self.val_metrics.update(y_probability.detach(), target)
        #     y_probability = torch.sigmoid(y_predict)
        #     y_predicts = y_probability >= 0.5
        #     for metric in self.val_metrics:
        #         if metric == 'val_AP' or 'val_AUROC':
        #             metric.update(y_predicts.detach(), target)
        #             self.val_metrics.update(y_predict[:, 1].detach(), target)
        #         else:
        #             self.val_metrics.update(y_probability.detach(), target)
        #
        # else:
        #     y_probability = torch.softmax(y_predict, dim=1)
        #     self.val_metrics.update(one_hot(y_probability.detach()), target)
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
        predictions, y_predict, y_repre = self.predict_batch(batch, preprocess=False)

        target = y.type(torch.LongTensor).cuda()

        test_loss = self.loss_fn(y_repre, target)
        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.test_metrics.update(y_predicts.detach(), target)
        # target = y.cuda()
        # target = torch.nn.functional.one_hot(target, num_classes=self.n_class)
        # BCELoss
        # m = torch.nn.Sigmoid()
        # test_loss = self.loss_fn(m(y_predict).squeeze(), target)
        # test_loss = self.loss_fn(y_repre, target)

        # fwd_predict, bwd_predict = self.predict_batch(batch, preprocess=False)
        # test_loss = self.loss_fn(fwd_predict, target)
        # test_loss += self.loss_fn(bwd_predict, target)

        # Logging
        # y_predict = fwd_predict
        # y_probability = y_predict[:, 1]
        # if self.n_class == 2:
        #     y_probability = torch.argmax(torch.sigmoid(y_predict), dim=1)
        #     self.test_metrics.update(y_probability.detach(), target)
        # else:
        #     y_probability = torch.softmax(y_predict, dim=1)
        #     self.test_metrics.update(one_hot(y_probability.detach()), target)

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss

