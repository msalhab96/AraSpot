from pathlib import Path
from typing import Union
from config import get_args
from models import get_model
from processors import get_speech_processor
from utils import load_json
from torch.nn import Module
import torch.nn.functional as F
from interfaces import IProcessor
import torch
from constants import path_column, class_column
import pandas as pd


def single_predict(
        device: str,
        file_path: Union[Path, str],
        speech_processor: IProcessor,
        model: Module
        ):
    x = speech_processor.process(file_path)
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    preds = F.softmax(model(x), dim=-1)
    cls = torch.argmax(preds).item()
    p = torch.max(preds).item()
    return cls, p


def evaluate(
        device: str,
        cls_mapper: dict,
        file_path: Union[Path, str],
        save_to: Union[Path, str],
        speech_processor: IProcessor,
        model: Module,
        pred_key='pred',
        prob_key='p'

        ):
    idx_to_cls = {
        value: key for key, value in cls_mapper.items()
        }
    df = pd.read_csv(file_path)
    results = df[path_column].apply(
        lambda x: single_predict(
            device=device,
            file_path=x,
            speech_processor=speech_processor,
            model=model
        )
        )
    df[pred_key] = [idx_to_cls[item[0]] for item in results]
    df[prob_key] = [item[1] for item in results]
    df.to_csv(save_to)
    acc = (df[pred_key] == df[class_column]).mean() * 100
    print(
        f'The accuracy on {file_path} is: {acc}, prediction results saved to {save_to}!'
        )


if __name__ == '__main__':
    cfg = get_args()
    mapper = load_json(cfg.cls_mapper)
    speech_processor = get_speech_processor(cfg)
    model = get_model(cfg, len(mapper))
    model = model.to(cfg.device).eval()
    cfg.eval_file
    evaluate(
        device=cfg.device,
        cls_mapper=mapper,
        file_path=cfg.eval_file,
        save_to=cfg.eval_file_result,
        speech_processor=speech_processor,
        model=model
        )
