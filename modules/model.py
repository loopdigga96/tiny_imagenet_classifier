from collections import OrderedDict

import torch
from torch import nn
import torch.functional as F
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class LightningTinyImageNetClassifier(pl.LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 loss_function: torch.nn.Module,
                 batch_size=64,
                 lr=1e-3):

        super(LightningTinyImageNetClassifier, self).__init__()
        self.model = backbone
        self.loss_function = loss_function

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.num_workers = 5
        self.lr = lr

    def setup_datasets(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, target = batch['image'], batch['label']
        output = self.model(images)
        loss = self.loss_function(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'acc1': acc1,
            'acc5': acc5,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, target = batch['image'], batch['label']
        output = self.model(images)
        loss_val = self.loss_function(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return output

    def test_step(self, batch, batch_idx):
        images, filename = batch['image'], batch['filename']
        output = self.model(images)
        pred = output.argmax(dim=1).reshape(-1)
        return {'predictions': pred, 'filename': filename}

    def inference(self):
        output_results = []
        for idx, batch in enumerate(self.test_dataloader()):
            res = self.test_step(batch, idx)
            output_results.append(res)
        filenames = []
        pred_tensors = []

        for out in output_results:
            filenames.extend(out['filename'])
            pred_tensors.append(out['predictions'])

        predictions = torch.cat(pred_tensors).tolist()
        return {'preds': predictions, 'filenames': filenames}

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            tqdm_dict[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict,
                  'val_loss': tqdm_dict["val_loss"], 'val_acc1': tqdm_dict['val_acc1']}

        return result

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=1,
                                                         min_lr=1e-6,
                                                         factor=0.3,
                                                         mode='min')
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k / batch_size)
                # res.append(correct_k.mul_(100.0 / batch_size))
            return res


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
