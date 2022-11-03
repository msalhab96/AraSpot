from abc import ABC, abstractmethod


class IProcessor(ABC):

    @abstractmethod
    def process():
        pass


class ILogger(ABC):
    
    @abstractmethod
    def log():
        pass
    
    @abstractmethod
    def record():
        pass