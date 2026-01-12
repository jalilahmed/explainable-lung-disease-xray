import numpy as np
import torch
import pandas as pd
import os
import pathlib
import random
import shutil
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

import sys
sys.path.append('.')

from src.datasets import NIHChestXrayDataset
from src.trainers import Trainer
from src.utils import MultiLabelAccuracy


#Fix seed
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

#device
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

#data lists
TRAIN_DF = pd.read_csv(os.path.join('/usr/app/data', "df_train.csv"))
VALID_DF = pd.read_csv(os.path.join('/usr/app/data', "df_val.csv"))


def run_experiment(experiment_dict):
    #Extract variables from experiment dictionary
    # Extract model from dictionary
    model = experiment_dict['model']
    
    # Experiment name and description
    experiment_name = experiment_dict['experiment_name']
    experiment_description=experiment_dict['experiment_description']
    
    # Loss function
    loss_fn = experiment_dict['loss_fn']

    # Metric function
    metric_fn = experiment_dict['metric_fn']

    # Hyperparameters
    learning_rate=experiment_dict['learning_rate']
    num_epochs=experiment_dict['num_epochs']
    batch_size=experiment_dict['batch_size']


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataloader
    train_dl = DataLoader(NIHChestXrayDataset(TRAIN_DF, transform=None),
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True)

    valid_dl = DataLoader(NIHChestXrayDataset(VALID_DF, transform=None),
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True)

    folder = './results/{}'.format(experiment_name)

    print("\n================Experiment-{}================\n".format(experiment_name))

    print('Training on Device: ' + DEVICE)

    folder = os.path.join(folder)
    os.makedirs(folder)
    pathlib.Path(os.path.join(folder, 'Readme.txt')).write_text(
        experiment_description)


    trainer = Trainer(model,
                      train_dl,
                      valid_dl,
                      num_epochs,
                      loss_fn,
                      metric_fn,
                      optimizer,
                      folder,
                      device=DEVICE)

    train_loss, valid_loss, train_metric, valid_metric = trainer.fit()

    df = pd.DataFrame({
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_metric": train_metric,
        "valid_metric": valid_metric,
    })

    model.to('cpu')
    torch.save(model.state_dict(), os.path.join(folder, 'last.pt'))

    df.to_csv(os.path.join(folder, '{}.csv'.format(experiment_name)))

    shutil.copy(__file__, os.path.join(folder, 'experiment_script.py'))

    return None


def main():
   
   ResNet = models.resnet50(pretrained=False)
   ResNet.fc = nn.Linear(ResNet.fc.in_features, 15)
   
   experiment_1_0 = {
        'experiment_name': 'Tutorial_27122025_00',
        'model': ResNet,
        'loss_fn': nn.BCEWithLogitsLoss(),
        'metric_fn': MultiLabelAccuracy(threshold=0.5, reduction='mean'),
        'input_size': (224, 224),
        'batch_size': 64, 
        'learning_rate': 1e-3,
        'num_epochs': 30,
        'experiment_description': 'ResNet50 with BCEWithLogitsLoss on NIH Chest X-ray dataset'
   }

   experiments = [experiment_1_0]
   
   for experiment_dict in experiments:
      _ = run_experiment(experiment_dict)


if __name__ == '__main__':
    main()