import torch

from . import Classifier
from ..nn import BiGRILC


class BritsClassifier(Classifier):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 n_class=2):
        super(BritsClassifier, self).__init__(model_class=model_class,
                                              model_kwargs=model_kwargs,
                                              optim_class=optim_class,
                                              optim_kwargs=optim_kwargs,
                                              loss_fn=loss_fn,
                                              metrics=metrics,
                                              scheduler_class=scheduler_class,
                                              scheduler_kwargs=scheduler_kwargs,
                                              )


        self.n_class = n_class

    def training_step(self, batch, batch_idx):
        # unpack_batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        y = batch_data.pop('y')
        # y = y[:, 0, -1]
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)
        # compute predictions and compute loss
        predictions, states, y_repre = self.predict_batch(batch, preprocess=False)
        target = y.type(torch.LongTensor).cuda()
        loss = self.loss_fn(y_repre.float(), target)


        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.train_metrics.update(y_predicts.detach(), target)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        y = batch_data.pop('y')
        # y = y[:, 0, -1]
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)

        # Compute predictions and loss
        predictions, y_repre = self.predict_batch(batch, preprocess=False)

        target = y.type(torch.LongTensor).cuda()
        val_loss = self.loss_fn(y_repre.float(), target)
        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.val_metrics.update(y_predicts.detach(), target)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return val_loss

    def test_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        y = batch_data.pop('y')
        # y = y[:, 0, -1]
        y = y[:, 0, -1, :]
        y = torch.squeeze(y)

        predictions, y_repre = self.predict_batch(batch, preprocess=False)

        target = y.type(torch.LongTensor).cuda()

        test_loss = self.loss_fn(y_repre, target)
        if self.n_class == 2:
            y_predicts = torch.softmax(y_repre, dim=1)[:, 1]
        else:
            y_predicts = torch.softmax(y_repre, dim=1)
        self.test_metrics.update(y_predicts.detach(), target)

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss