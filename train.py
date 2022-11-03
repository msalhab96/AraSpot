import os
import torch
from utils import calc_acc


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

    def fit(self):
        for epoch in range(self.epochs):
            self.train()
            self.test()
            self.logger.log()
            if self.is_terminate() is True:
                print('The model is not improving any more!')
                break
            else:
                self.export_ckpt(epoch)

    def export_ckpt(self, epoch: int):
        path = os.path.join(self.outdir, f'checkpoint_{epoch}.pt')
        torch.save(self.model.state_dict(), path)

    def train(self):
        self.model.train()
        total_loss = 0
        preds = []
        targets = []
        for batch in self.train_loader:
            [speech, text, mask, cls] = batch
            speech = speech.to(self.device)
            text = text.to(self.device)
            mask = mask.to(self.device)
            cls = cls.to(self.device)
            self.optimizer.zero_grad()
            out, preds, result = self.model(speech, text)
            loss = self.criterion(
                out, preds, result, text, mask, cls
                )
            loss.backward()
            self.optimizer.step()
            preds.append(torch.argmax(preds.cpu(), dim=-1))
            targets.append(cls.cpu())
            total_loss += loss.item()
        self.logger.record(
            'train_loss', total_loss / len(self.train_loader)
            )
        self.logger.record(
            'train_acc', calc_acc(
                torch.vstack(preds),
                torch.vstack(targets)
                )
            )

    def test(self):
        self.model.eval()
        total_loss = 0
        preds = []
        targets = []
        for batch in self.test_loader:
            [speech, text, mask, cls] = batch
            speech = speech.to(self.device)
            text = text.to(self.device)
            mask = mask.to(self.device)
            cls = cls.to(self.device)
            out, preds, result = self.model(speech, text)
            loss = self.criterion(
                out, preds, result, text, mask, cls
                )
            preds.append(torch.argmax(preds.cpu(), dim=-1))
            targets.append(cls.cpu())
            total_loss += loss.item()
        self.logger.record(
            'test_loss', total_loss / len(self.train_loader)
            )
        self.logger.record(
            'test_acc', calc_acc(
                torch.vstack(preds),
                torch.vstack(targets)
            )
        )
