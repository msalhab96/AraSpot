from argparse import ArgumentParser
from torch import Tensor
from typing import Union
from pathlib import Path
from constants import path_column, class_column
import pandas as pd
import torchaudio
import random
import torch
import os


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--n_examples', nargs='+', type=int, required=True,
        help='The number of examples per each file provided in --files'
        )
    parser.add_argument(
        '--key', type=str, default='silence'
    )
    parser.add_argument(
        '--files', nargs='+', type=str, required=True,
        help='the csv files to add the examples to'
    )
    parser.add_argument(
        '--noise_dir', type=str, default='background_noise'
    )
    parser.add_argument(
        '--save_to', type=str, default='dataset/noise'
    )
    parser.add_argument(
        '--length', type=int, default=16000
    )
    parser.add_argument(
        '--sr', type=int, default=16000
    )
    return parser.parse_args()


def load_audios(args):
    audios = []
    for file in os.listdir(args.noise_dir):
        path = os.path.join(args.noise_dir, file)
        x, _ = torchaudio.load(path)
        audios.append(x)
    audios = torch.hstack(audios)
    return audios


def generate_audio(
        args,
        audios: Tensor,
        file_path: Union[str, Path]
        ) -> None:
    length = audios.shape[1]
    start = random.randint(0, length - args.length)
    end = start + args.length
    noise = audios[:, start: end]
    torchaudio.save(
        filepath=file_path,
        src=noise,
        sample_rate=args.sr,
        channels_first=True
        )


def generate(
        args,
        audios: Tensor,
        n_examples: int,
        key: str,
        save_to: str,
        file_path: str,
        pref: str
        ):
    df = pd.read_csv(file_path)
    for i in range(n_examples):
        aud_path = os.path.join(
            save_to, f'{key}_{pref}_{i}.wav'
            )
        generate_audio(args, audios, aud_path)
        df = df.append(
            {
                path_column: aud_path,
                class_column: key
                },
            ignore_index=True
            )
    df.to_csv(file_path)
    print(
        f'{n_examples} examples added to {file_path}!'
        )


def main():
    args = get_args()
    if os.path.exists(args.save_to) is False:
        os.mkdir(args.save_to)
    audios = load_audios(args)
    iterator = enumerate(zip(args.files, args.n_examples))
    for i, (file_path, n_examples) in iterator:
        generate(
            args=args,
            audios=audios,
            n_examples=n_examples,
            key=args.key,
            save_to=args.save_to,
            file_path=file_path,
            pref=i
            )


if __name__ == '__main__':
    main()
