from torch.optim import Adam


class AdamLinearDecay:
    def __init__(
            self,
            parameters,
            lr: float,
            epochs: int,
            *args,
            **kwargs
            ):
        self.epochs = epochs
        self.lr = lr
        self._counter = 0
        self.optimizer = Adam(
            parameters,
            lr=self.get_lr()
        )

    def get_lr(self) -> float:
        return self.lr * (1 - (self._counter/self.epochs))

    def update_lr(self) -> None:
        self._counter += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, counter) -> None:
        self.optimizer.load_state_dict(state_dict)
        self._counter = counter
