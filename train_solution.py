from modules import LightningTinyImageNetClassifier, TinyImagenetDataset, TinyImagenetTestSet, LabelSmoothingLoss

from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torchvision import models
from torch.nn.modules import loss
from efficientnet_pytorch import EfficientNet

pl.seed_everything(42)
batch_size = 36
epochs = 50
lr = 1e-4
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = "test"
num_of_classes = 200

train_transform = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


def get_backbone():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(1792, num_of_classes)
    return model


def main():
    results_root = Path("./results/") / f"{experiment_name}_{timestamp}"
    checkpoints = results_root / "checkpoints"
    tensorboard_logs = results_root

    # setting up dataset
    data_path = Path("./data")
    data_path.mkdir(exist_ok=True)
    data_root = data_path / "tiny-imagenet-200"
    train_path = data_root / "train"
    val_path = data_root / "val"
    test_path = data_root / "test"

    all_folders = [
        dir_name
        for r, d, f in os.walk(train_path)
        for dir_name in d
        if dir_name != "images"
    ]
    folders_to_num = {val: index for index, val in enumerate(all_folders)}

    # LABELS = pd.read_csv(
    #     data_root / "words.txt", sep="\t", header=None, index_col=0)[1].to_dict()
    val_labels = pd.read_csv(
        data_root / "val" / "val_annotations.txt", sep="\t", header=None, index_col=0)[1].to_dict()

    train_dataset = TinyImagenetDataset(train_path, folders_to_num, val_labels, train_transform)
    val_dataset = TinyImagenetDataset(val_path / "images", folders_to_num, val_labels, test_transform)
    test_dataset = TinyImagenetTestSet(test_path / "images", test_transform)

    model = get_backbone()

    loss_function = LabelSmoothingLoss(classes=num_of_classes, smoothing=0.1)

    model_checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=os.path.join(f'{checkpoints}/', '{epoch}-{val_loss:.2f}'),
        monitor='val_loss',
        verbose=True,
        save_top_k=1,
        mode='min'
    )
    early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=4, verbose=True, mode='min')

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=tensorboard_logs, name='', version='tb_logs'
    )
    pl_model = LightningTinyImageNetClassifier(model, loss_function, batch_size)
    pl_model.setup_datasets(train_dataset, val_dataset, test_dataset)
    trainer_args = {'max_epochs': epochs,
                    'checkpoint_callback': model_checkpoint,
                    'logger': [tb_logger],
                    'early_stop_callback': early_stop}

    if torch.cuda.is_available():
        trainer_args['gpus'] = [1]

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(pl_model)
    best_checkpoint_name = os.listdir(checkpoints)[-1]
    full_checkpoint_path = os.path.join(checkpoints, best_checkpoint_name)

    # using standart torch load because lightning have some bug in its implementation
    best_model = LightningTinyImageNetClassifier(get_backbone(), loss_function, batch_size, lr)
    best_model.setup_datasets(train_dataset, val_dataset, test_dataset)
    best_model.load_state_dict(torch.load(full_checkpoint_path)['state_dict'])
    result = best_model.inference()

    pd.DataFrame(result, index=None).to_csv(os.path.join(results_root, 'test_preds.csv'))


if __name__ == '__main__':
    main()
