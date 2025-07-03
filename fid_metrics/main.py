import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
import os
import piq
import tqdm
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    calculate_fid,
    is_image_dir_path,
    is_video_path,
    postprocess_i2d_pred,
)


def build_loaders(type, paths, cfg):
    dls = []
    for path in paths:
        bs = cfg.batch_size
        dataset_cfgs = cfg.get('dataset')

        if is_video_path(path):
            if type == 'fid':
                if dataset_cfgs:
                    dataset_cfgs = dict(dataset_cfgs)
                    dataset_cfgs['sequence_length'] = bs
                else:
                    dataset_cfgs = {'sequence_length': bs}
                bs = 1
            C = VideoDataset
        elif is_image_dir_path(path, 'png'):
            C = ImageDataset if type == 'fid' else ImageSequenceDataset
        else:
            raise NotImplementedError

        dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
        dl = torch.utils.data.DataLoader(dataset, bs, shuffle=False, num_workers=cfg.num_workers)
        dls.append(dl)
    return dls


def build_model(type, cfg):
    if type == 'fid':
        return build_inception(cfg.dims)
    elif type == 'fvd':
        return build_inception3d(cfg.type, cfg.path)
    else:
        raise NotImplementedError


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for metric_cfgs in cfg.metrics:
        type = metric_cfgs.type
        dls = build_loaders(type, cfg.paths, metric_cfgs.data)
        model = build_model(type, metric_cfgs.model).to(device).eval()

        feats = [[], []]
        T = [[], []]
        for i, dl in enumerate(dls):
            if cfg.get('num_iters'):
                seq = range(cfg.num_iters // metric_cfgs.data.batch_size)
            else:
                seq = range(len(dl))
            dl = iter(dl)

            for _ in track(seq, description=f'{type}_{i}'):
                x = next(dl).to(device)
                if type == 'fid' and x.dim() == 5:
                    x = x.squeeze(0).transpose(0, 1)
                elif type == 'fvd':
                    x = x * 2 - 1  # [-1, 1]
                if type == 'fid':
                    T[i].append(x.clamp(0, 1).cpu())
                with torch.no_grad():
                    if type == 'fid':
                        pred = model(x)
                        pred = postprocess_i2d_pred(pred)
                    elif type == 'fvd':
                        if metric_cfgs.model.type == 'styleganv':
                            pred = model(x, return_features=True)
                        else:
                            pred = model(x)
                feats[i].append(pred.cpu().numpy())
            feats[i] = np.concatenate(feats[i], axis=0)
        fid = calculate_fid(*feats)
        print(f'{type.upper()}: {fid}')
        if type == 'fid':
            total = 0
            for j in tqdm.tqdm(range(len(T[0])), desc='LPIPS'):
                lpipschitz = piq.LPIPS()(T[0][j].cuda(), T[1][j].cuda())
                total += lpipschitz.item()
            lpipschitz = total / len(T[0])
            print(f'LPIPS: {lpipschitz}')
            T[0] = torch.cat(T[0], dim=0)
            T[1] = torch.cat(T[1], dim=0)
            ssim = piq.ssim(T[0], T[1], data_range=1.0)
            print(f'SSIM: {ssim}')
            psnr = piq.psnr(T[0], T[1], data_range=1.0)
            print(f'PSNR: {psnr}')


if __name__ == '__main__':
    main()
