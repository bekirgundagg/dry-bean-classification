# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the trainer object that would be used for
# !     training the given model using the given optimizer and loss
# !     function.
# !
# !     You need to implement the train_loop and eval_loop functions.
# !     You can also use additional packages in this file.
# !     If necessary, you can also add another functions as required.
# !
# ! Make sure that the final implementation is compatible with the
# !     other classes.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Dict
import wandb
from sklearn.metrics import f1_score
class Trainer:
    def __init__(self, model, optimizer, loss_fn, project_name, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.project_name = project_name
        self.device = device

        self.__dict__.update(kwargs)
        self.model = self.model.to(self.device)

    # You can update the following train function to apply regulation during the training process
    def train(self, training_data: DataLoader, validation_data: DataLoader, epochs: int):
        with wandb.init(project=self.project_name) as run:
            for epoch in range(epochs):
                train_result = self.train_loop(training_data)
                eval_result  = self.eval_loop(validation_data)

                self.log_wandb(run, epoch, train_result, eval_result)

    ######################################################################################
    # ! Update Trainer functions (train_loop, eval_loop and log_wandb)
    ######################################################################################
    # Make sure that your output of this function is a dictionary presenting the results of
    # performance metrics for single training loop
    def train_loop(self, data: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_f1 = 0.0
        for X,y in data:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_f1 += f1_score(y.argmax(dim=1).cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
        avg_loss = total_loss / len(data)
        avg_f1 = total_f1 / len(data)
        return {"loss": avg_loss, "f1": avg_f1}
    # Make sure that your output of this function is a dictionary presenting the results of
    # performance metrics for single evaluation loop
    def eval_loop(self, data: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        with torch.no_grad():
            for X,y in data:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)

                total_loss += loss.item()
                total_f1 += f1_score(y.argmax(dim=1).cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
        total_loss = total_loss / len(data)
        total_f1 = total_f1 / len(data)
        return {"loss": total_loss, "f1": total_f1}

    def log_wandb(self, run, epoch, training_result, eval_result):
        run.log({
            "epoch": epoch,
            "train_loss": training_result["loss"],
            "train_f1": training_result["f1"],
            "eval_loss": eval_result["loss"],
            "eval_f1": eval_result["f1"]
        })
