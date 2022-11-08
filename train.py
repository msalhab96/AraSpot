import math
import os
import torch
from utils import calc_acc, load_json


class Trainer:
    def __init__(
            self,
            train_loader,
            test_loader,
            model,
            optimizer,
            criterion,
            outdir,
            epochs,
            device,
            logger
            ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.outdir = outdir
        self.epochs = epochs
        self.device = device
        self.logger = logger
        self._loss = math.inf
        self._last = math.inf

    def is_terminate(self):
        # TODO
        return False

    def fit(self):
        for epoch in range(self.epochs):
            self.train()
            self.test()
            self.logger.log()
            if self._last < self._loss:
                self._loss = self._last
                self.export_ckpt(epoch)

    def export_ckpt(self, epoch: int):
        path = os.path.join(self.outdir, f'checkpoint_{epoch}.pt')
        print(f'checkpoint {path} saved!')
        print('=' * 50)
        torch.save(self.model.state_dict(), path)

    def train(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        for batch in self.train_loader:
            [speech, cls] = batch
            speech = speech.to(self.device)
            cls = cls.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(speech)
            loss = self.criterion(preds, cls)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.logger.record(
            'train_loss', total_loss / len(self.train_loader)
            )
        for batch in self.train_loader:
            [speech, cls] = batch
            speech = speech.to(self.device)
            cls = cls.to(self.device)
            preds = self.model(speech)
            all_preds.append(torch.argmax(preds.cpu(), dim=-1))
            all_targets.append(cls.cpu())
        self.logger.record(
            'train_acc', calc_acc(
                torch.hstack(all_preds),
                torch.hstack(all_targets)
                )
            )

    @torch.no_grad()
    def test(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        for batch in self.test_loader:
            [speech, cls] = batch
            speech = speech.to(self.device)
            cls = cls.to(self.device)
            preds = self.model(speech)
            loss = self.criterion(preds, cls)
            all_preds.append(torch.argmax(preds.cpu(), dim=-1))
            all_targets.append(cls.cpu())
            total_loss += loss.item()
        self.logger.record(
            'test_loss', total_loss / len(self.train_loader)
            )
        self.logger.record(
            'test_acc', calc_acc(
                torch.hstack(all_preds),
                torch.hstack(all_targets)
            )
        )
        self._last = total_loss
