from datetime import datetime
import torch
import os
import numpy as np
import json


class Trainer(object):
    """Trainer class for model training and validation
    """
    def __init__(self, model, train_dl, valid_dl, 
                 num_epochs, 
                 loss_fn,
                 metric_fn,
                 optimizer,
                 folder,
                 device='cpu',
                 scheduler=None,
                 flatline_patience=10):
        """Initialize the Trainer with model, data loaders, and training parameters.
        Args:
            model (torch.nn.Module): The model to be trained.
            train_dl (DataLoader): DataLoader for training data.
            valid_dl (DataLoader): DataLoader for validation data.
            num_epochs (int): Number of epochs to train.
            loss_fn (function): Loss function.
            metric_fn (function): Metric function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            folder (str): Folder to save logs and checkpoints.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            flatline_patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 6.
        """
        
        self.device = device
        self.model = model.to(self.device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.folder = folder

        self.flatline_patience = flatline_patience

        self.no_change_counter = 0
        self.prev_loss = None
        self.prev_metric = None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [INFO] Trainer in normal mode")
        
        self.log_file = os.path.join(folder, "logs.txt")

    def log(self, message):
        """Log a message to the log file with a timestamp.
        Args:
            message (str): Message to log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(msg)
    
    def forward(self, batch):
        """Forward pass for a single batch.
        Args:
            batch (tuple): A batch of data (inputs, labels).
        Returns:
            tuple: Computed loss and metric for the batch.
        """
        x = batch[0].to(self.device)
        y = batch[1].to(self.device).float()

        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        metric = self.metric_fn(y_hat, y)

        return loss, metric

    def train_loop(self):
        """Training loop for one epoch.
        Returns:
            tuple: Average loss and metric for the epoch.
        """
        
        self.model.train()

        losses = []
        metrics = []

        for i, batch in enumerate(self.train_dl):

            self.optimizer.zero_grad()

            loss, metric = self.forward(batch)

            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())
            metrics.append(metric.item())
            
        return np.mean(losses), np.mean(metrics)    

    def valid_loop(self):
        """Validation loop for one epoch.
        Returns:
            tuple: Average loss and metric for the epoch.
        """
        self.model.eval()

        losses = []
        metrics = []

        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):

                loss, metric = self.forward(batch)

                losses.append(loss.item())
                metrics.append(metric.item())

        return np.mean(losses), np.mean(metrics)
    
    def epoch_summary(self, epoch, train_loss, valid_loss, train_metric, valid_metric):
        """Log the summary of an epoch.
        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            valid_loss (float): Validation loss for the epoch.
        """
        msg = (
            f"Epoch {epoch+1:03d}/{self.num_epochs}: "
            f"Train Loss={train_loss:.6f}, Val Loss={valid_loss:.6f} |"
            f" Train Metric={train_metric:.6f}, Val Metric={valid_metric:.6f}"
        )
        self.log(msg)

    def fit(self):
        """Fit the model using the training and validation loops.
        Returns:
            dict: History of training and validation losses
        """

        self.log("Training started")
        
        history = {
            "train_loss": [],
            "valid_loss": [], 
            "train_metric": [],
            "valid_metric": []
        }
        
        for epoch in range(self.num_epochs):
            tLoss, tMetric = self.train_loop()
            vLoss, vMetric = self.valid_loop()

            self.epoch_summary(epoch, tLoss, vLoss, tMetric, vMetric)

            if self.scheduler is not None:
                self.scheduler.step(vLoss)

            history["train_loss"].append(tLoss)
            history["train_metric"].append(tMetric)
            history["valid_loss"].append(vLoss)
            history["valid_metric"].append(vMetric)

            if epoch % 50 == 0 or epoch == 0.:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                torch.save(self.model.state_dict(), os.path.join(self.folder, f'{now}_epoch_{epoch}.pt'))

            if vLoss <= np.min(history["valid_loss"]):
                self.log(f"Saving new minimum loss checkpoint at epoch {epoch+1} with val loss {vLoss:.6f}")
                torch.save(self.model.state_dict(), os.path.join(self.folder, f'checkpoint_min_loss.pt'))

            if vMetric >= np.max(history["valid_metric"]):
                self.log(f"Saving new maximum metric checkpoint at epoch {epoch+1} with val metric {vMetric:.6f}")
                torch.save(self.model.state_dict(), os.path.join(self.folder, f'checkpoint_max_metric.pt'))

            if self.prev_loss is not None and abs(self.prev_loss - vLoss) < 1e-6:
                self.no_change_counter += 1
                if self.no_change_counter >= self.flatline_patience:
                    self.log(f"[ALERT] Detected flatline for {self.flatline_patience} epochs.")
                    break
            else:
                self.no_change_counter = 0
                
            self.prev_loss = vLoss
            self.prev_metric = vMetric

        self.log(f"Training complete")
        
        return history