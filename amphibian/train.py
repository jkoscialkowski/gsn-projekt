# TODO 1: Implement random search
# TODO 2: Look if parameters are passed correctly

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


class SingleTrainer:
    def __init__(self, model, batch_size, max_epochs=500,
                 early_stopping_patience=None):
        """
        Class SingleTrainer -

        :param batch_size: size of the batch
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=model.parameters())

    def train(self, train_ds, valid_ds):
        dl = DataLoader(train_ds, batch_size=self.batch_size,
                        shuffle=True, num_workers=8)
        losses = []
        for epoch in range(self.max_epochs):
            losses.append({'train_loss': [], 'valid_loss': []})
            for idx_batch, batch in enumerate(dl):
                # Switch to training mode
                self.model.train()
                out = self.model(batch['observations'].permute(1,0,2))
                tr_loss = self.loss(out, batch['y'])
                losses[epoch]['train_loss'].append(tr_loss.item())
                tr_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Switch to evaluation mode
                self.model.eval()
                val_loss = self.loss(
                    self.model(valid_ds.observations),
                    valid_ds.y
                )
                losses[epoch]['valid_loss'].append(val_loss)
                if self.early_stopping_patience:
                    es_threshold = max(losses[epoch]['valid_loss'][
                        (idx_batch-self.early_stopping_patience):idx_batch
                    ])
                    if val_loss > es_threshold:
                        break