from time import sleep
import torch
from tqdm import tqdm
from ml.models import LinearModel
from ml.data import load_mnist_data
from ml.utils import set_device

from celery import Celery

import time
import os
from sklearn.model_selection import ParameterGrid

app = Celery('train', backend=os.environ.get("CELERY_RESULT_BACKEND_URL", "redis://127.0.0.1:6379"), broker=os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379"))
app.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379")


class Trainer:
    def __init__(self, model, learning_rate=None, optimizer=None, criterion=None, device=None):
        """Initialize the trainer"""
        self.model = model

        self.lr = learning_rate if learning_rate else .001

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.model = self.model.to(device)

    def get_model(self):
        return self.model

    def train(self, num_epochs, train_dataloader, val_dataloader=None):
        """Trains the model and logs the results"""
        # Set result dict
        results = {"train_loss": [], "train_acc": []}
        if val_dataloader is not None:
            results["val_loss"] = []
            results["val_acc"] = []

        # Start training
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self.train_epoch(
                dataloader=train_dataloader)
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            # Validate only if we have a val dataloader
            if val_dataloader is not None:
                val_loss, val_acc = self.eval_epoch(dataloader=val_dataloader)
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
            
        return results

    def train_epoch(self, dataloader):
        """Trains one epoch"""
        self.model.train()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Train step
            self.optimizer.zero_grad()  # Clear gradients.
            outs = self.model(X)  # Perform a single forward pass.
            loss = self.criterion(outs, y)

            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs,
                                       dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Eval
            outs = self.model(X)
            loss = self.criterion(outs, y)

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs,
                                       dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc


@app.task()
def train(drop_out: float, num_epochs: int, learning_rate: float):
    device = set_device()

    # Prepare for training
    print("Loading data...")
    train_dataloader, val_dataloader = load_mnist_data()

    # Train
    print("Training model")
    model = LinearModel(drop_out).to(device)

    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    """Trains the model and logs the results"""
    # Set result dict
    results = {"train_loss": [], "train_acc": []}
    if val_dataloader is not None:
        results["val_loss"] = []
        results["val_acc"] = []

    # Start training
    start = time.time()
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = trainer.train_epoch(
            dataloader=train_dataloader)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        # Validate only if we have a val dataloader
        if val_dataloader is not None:
            val_loss, val_acc = trainer.eval_epoch(dataloader=val_dataloader)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

        train.update_state(state='PROGRESS',
            meta={'current': epoch + 1, 
                  'total': num_epochs,
                  'metrics': results})
    end = time.time()
    
    results["run_time"] = end - start

    return results

def train_grid_search(drop_out: float, num_epochs: int, learning_rate: float):
    device = set_device()

    # Prepare for training
    print("Loading data...")
    train_dataloader, val_dataloader = load_mnist_data()

    # Train
    print("Training model")
    model = LinearModel(drop_out).to(device)

    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    """Trains the model and logs the results"""
    # Set result dict
    results = {"train_loss": [], "train_acc": []}
    if val_dataloader is not None:
        results["val_loss"] = []
        results["val_acc"] = []

    # Start training
    start = time.time()
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = trainer.train_epoch(
            dataloader=train_dataloader)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        # Validate only if we have a val dataloader
        if val_dataloader is not None:
            val_loss, val_acc = trainer.eval_epoch(dataloader=val_dataloader)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

    return results



@app.task()
def grid_search(params_dict):
    param_grid = ParameterGrid(params_dict)
    count = 1
    best_params = {}
    best_acc = 0

    start = time.time()
    for dict_ in param_grid:
        results = train_grid_search(drop_out=dict_['drop_out'], num_epochs=dict_['epochs'], learning_rate=dict_['learning_rate'])
        if results['val_acc'][-1] > best_acc:
            best_acc = results['val_acc'][-1]
            best_params = dict_
        grid_search.update_state(state='PROGRESS',
                meta={'current': count, 
                      'total': len(param_grid)
                }
        )
        count += 1
    end = time.time()
    
    return {"best_params": best_params, "best_acc": best_acc, "run_time": end - start}


