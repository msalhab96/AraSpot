from argparse import ArgumentParser


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument(
        '--residual_scaler', type=float, default=0.5
        )
    group.add_argument(
        '--kernel_size', type=int, default=5
        )
    group.add_argument(
        '--scaling_factor', type=int, default=2
    )
    group.add_argument(
        '--emb_dim', type=int, default=64
    )
    group.add_argument(
        '--n_layers', type=int, default=1
    )
    group.add_argument(
        '--enc_dim', type=int, default=128
    )
    group.add_argument(
        '--bidirectional', type=bool, default=True
    )
    group.add_argument(
        '--p_dropout', type=float, default=0.25
    )
    group.add_argument(
        '--ckpt_path', type=str, default=''
    )
    group.add_argument(
        '--h', type=int, default=4
    )

def add_aug_args(parser):
    group = parser.add_argument_group('augmentation')
    group.add_argument(
        '--max_freq_len', type=int, default=5
        )
    group.add_argument(
        '--max_time_len', type=int, default=5
        )
    group.add_argument(
        '--n_mask', type=int, default=5
        )
    group.add_argument(
        '--noise_path', type=str, default='noise/'
        )
    group.add_argument(
        '--p_aug', type=float, default=0.5
        )


def add_feature_args(parser):
    group = parser.add_argument_group('Speech Features')
    group.add_argument(
        '--sample_rate', type=int, default=16000
        )
    group.add_argument(
        '--n_mfcc', type=int, default=40
        )
    group.add_argument(
        '--feature', type=str, default='mfcc'
        )
    group.add_argument(
        '--n_fft', type=int, default=400
        )
    group.add_argument(
        '--win_length', type=int, default=340
        )
    group.add_argument(
        '--hop_length', type=int, default=120
        )
    group.add_argument(
        '--n_mels', type=int, default=80
        )


def add_train_args(parser):
    group = parser.add_argument_group('training')
    group.add_argument(
        '--epochs', type=int, default=200
        )
    group.add_argument(
        '--lr', type=float, default=0.005
        )
    group.add_argument(
        '--batch_size', type=int, default=32
        )
    group.add_argument(
        '--alpha', type=float, default=0.5
        )
    group.add_argument(
        '--outdir', type=str, default='outdir/'
        )
    group.add_argument(
        '--device', type=str, default='cuda:1'
        )
    group.add_argument(
        '--logdir', type=str, default='outdir/logs'
        )
    group.add_argument(
        '--train_path', type=str, default='train.csv'
        )
    group.add_argument(
        '--test_path', type=str, default='val.csv'
        )
    group.add_argument(
        '--chars_mapper', type=str, default='mappers/chars.json'
        )
    group.add_argument(
        '--cls_mapper', type=str, default='mappers/cls.json'
        )
    group.add_argument(
        '--text_mapper', type=str, default='mappers/mapper.json'
        )
    group.add_argument(
        '--max_len', type=str, default=50
        )


def get_args() -> dict:
    parser = ArgumentParser()
    add_aug_args(parser)
    add_model_args(parser)
    add_feature_args(parser)
    add_train_args(parser)
    return parser.parse_args()


def get_model_args(cfg, n_classes) -> dict:
    return {
        'feat_size': cfg.n_mfcc if cfg.feature == 'mfcc' else cfg.n_mels,
        'enc_dim': cfg.enc_dim,
        'kernel_size': cfg.kernel_size,
        'h': cfg.h,
        'n_classes': n_classes,
        'n_layers': cfg.n_layers,
        'bidirectional': cfg.bidirectional,
        'scaling_factor': cfg.scaling_factor,
        'residual_scaler': cfg.residual_scaler,
        'p_dropout': cfg.p_dropout,
        'device': cfg.device
    }


def get_feat_args(cfg):
    mel = {
        'n_fft': cfg.n_fft,
        'win_length': cfg.win_length,
        'hop_length': cfg.hop_length,
        'n_mels': cfg.n_mels
        }
    if cfg.feature == 'mfcc':
        return {
            'sample_rate': cfg.sample_rate,
            'melkwargs': mel
        }
    return dict(
        **{'sample_rate': cfg.sample_rate,},
        **mel
        )