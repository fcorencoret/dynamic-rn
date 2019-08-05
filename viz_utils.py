import sys
import json
import pickle
import argparse
from collections import OrderedDict

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import numpy as np

import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt

from attrdict import AttrDict
from tqdm import tqdm_notebook as tqdm

from model import RN
from utils import build_dictionaries, collate_samples_from_pixels
from train import initialize_dataset
from clevr_dataset_connector import ClevrDataset
from viz.IntermediateLayerGetter import IntermediateLayerGetter, rgetattr

_mha_return_layers = {
    'rl.mha_fc1': 'mha_fc1',
    'rl.mha_fc2': 'mha_fc2',
    'rl.mha_fc3': 'mha_fc3',

    'rl.mha_layers.0': 'mha_gc0',
    'rl.mha_layers.1': 'mha_gc1',
    'rl.mha_layers.2': 'mha_gc2',
    'rl.mha_layers.3': 'mha_gc3',
}

_identity_return_layers = {
    'rl.identity_fc1': 'identity_fc1',
    'rl.identity_fc2': 'identity_fc2',
    'rl.identity_fc3': 'identity_fc3',
    

    'rl.identity_layers.0': 'identity_gc0',
    'rl.identity_layers.1': 'identity_gc1',
    'rl.identity_layers.2': 'identity_gc2',
    'rl.identity_layers.3': 'identity_gc3',
}

# return_layers = {
#     **_mha_return_layers,
#     **_identity_return_layers,
# }

g_keys = [
    'identity_gc0',
    'mha_gc1',
    'identity_gc1',
    'mha_gc2',
    'identity_gc2',
    'mha_gc3',
    'identity_gc3',
]
f_keys = [
    'mha_fc1',
    'identity_fc1',
    'mha_fc2',
    'identity_fc2',
    'mha_fc3',
    'identity_fc3'
]

mha_keys = [
    'mha_gc1',
    'mha_gc2',
    'mha_gc3',
    'mha_fc1',
    'mha_fc2',
    'mha_fc3',

]

identity_keys = [
    'identity_gc0',
    'identity_gc1',
    'identity_gc2',
    'identity_gc3',
    'identity_fc1',
    'identity_fc2',
    'identity_fc3',
]
 
def get_qtypes(dataset):
    return np.unique([q['program'][-1]['function'] for q in dataset.questions])

def get_indexes_per_qtype(qtypes, dataset):
    return {
        qtype: [index for index, q in enumerate(dataset.questions) if q['program'][-1]['function'] == qtype] for qtype in qtypes
    }

def get_datasets_per_qtype(dataset, idxs_per_qtype):
    return {
        qtype: Subset(dataset, idxs) for qtype, idxs in idxs_per_qtype.items()
    }

def get_selected_ds_per_qtype(ds, selected_idxs_per_qtype):
    return {
        qtype: Subset(ds, idxs) for qtype, idxs in selected_idxs_per_qtype.items()
    }

def stretch_tensor(t, size=256):
    if t.size(-1) == size:
        return t
    
    new_t = torch.ones(t.size(0), size, dtype=t.dtype)
    n_reps = size // t.size(-1)
    padding = (size % t.size(-1)) // 2
    
    for i, val in enumerate(t.t()):
        new_t[:, padding + i * n_reps: padding + (i + 1) * n_reps] = val.unsqueeze(-1).expand(t.size(0), n_reps)
        
    return new_t

def make_res_dict():
    return OrderedDict(
        identity_gc0=torch.empty(0, 64, 64, 256, dtype=torch.float32),

        mha_gc1=torch.empty(0, 256, dtype=torch.float32),
        identity_gc1=torch.empty(0, 64, 64, 256, dtype=torch.float32),

        mha_gc2=torch.empty(0, 256, dtype=torch.float32),
        identity_gc2=torch.empty(0, 64, 64, 256, dtype=torch.float32),

        mha_gc3=torch.empty(0, 256, dtype=torch.float32),
        identity_gc3=torch.empty(0, 64, 64, 256, dtype=torch.float32),

        mha_fc1=torch.empty(0, 256, dtype=torch.float32),
        identity_fc1=torch.empty(0, 256, dtype=torch.float32),

        mha_fc2=torch.empty(0, 256, dtype=torch.float32),
        identity_fc2=torch.empty(0, 256, dtype=torch.float32),

        mha_fc3=torch.empty(0, 28, dtype=torch.float32),
        identity_fc3=torch.empty(0, 28, dtype=torch.float32),
    )

def compute_attention_only(model, ds_per_qtype, bsz=32, samples_per_qtype=None, device=None):

    qtypes = list(ds_per_qtype.keys())
    model.eval()
    output = {}

    with torch.no_grad():
        for qtype in tqdm(qtypes):
            qres = make_res_dict()

            if samples_per_qtype is not None:
                ds = Subset(ds_per_qtype[qtype], np.arange(samples_per_qtype))
            else:
                ds = ds_per_qtype[qtype]

            loader = DataLoader(
                ds,
                batch_size=bsz,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_samples_from_pixels,
                )

            for i, b in tqdm(enumerate(loader), total=len(loader)):

                

                if device:
                    b['image'] = b['image'].to(device)
                    b['question'] = b['question'].to(device)

                mid_res, _ = model(b['image'], b['question'])

                if with_attention:
                    for k in mha_keys:
                        qres[k] = torch.cat((qres[k], mid_res[k][1].squeeze(1).cpu()))

                if with_identity:
                    for k in identity_keys:
                        if 'gc' in k:
                            qres[k] = torch.cat((qres[k], mid_res[k].view(-1, 64, 64, 256).cpu()))
                        else:
                            qres[k] = torch.cat((qres[k], mid_res[k].cpu()))

            output[qtype] = qres

    return output

def compute_mid_results(model, ds_per_qtype, with_attention=True, with_identity=True,
                        bsz=32, samples_per_qtype=None, device=None,
    ):

    qtypes = list(ds_per_qtype.keys())

    return_layers = {}
    if with_identity:
        return_layers.update(_identity_return_layers)
    if with_attention:
        return_layers.update(_mha_return_layers)

    model = IntermediateLayerGetter(model, return_layers, keep_output=False)
    model.eval()

    output = {}

    with torch.no_grad():
        for qtype in tqdm(qtypes):
            qres = make_res_dict()

            if samples_per_qtype is not None:
                ds = Subset(ds_per_qtype[qtype], np.arange(samples_per_qtype))
            else:
                ds = ds_per_qtype[qtype]

            loader = DataLoader(
                ds,
                batch_size=bsz,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_samples_from_pixels,
                )

            for i, b in tqdm(enumerate(loader), total=len(loader)):
                model._model.coord_tensor = None
                if device:
                    b['image'] = b['image'].to(device)
                    b['question'] = b['question'].to(device)

                mid_res, _ = model(b['image'], b['question'])

                if with_attention:
                    for k in mha_keys:
                        qres[k] = torch.cat((qres[k], mid_res[k][1].squeeze(1).cpu()))

                if with_identity:
                    for k in identity_keys:
                        if 'gc' in k:
                            qres[k] = torch.cat((qres[k], mid_res[k].view(-1, 64, 64, 256).cpu()))
                        else:
                            qres[k] = torch.cat((qres[k], mid_res[k].cpu()))

            output[qtype] = qres

    return output

def _get_mask_metrics(t):
    # t es el tensor con máscaras NO AGREGADAS

    histograms_mean = torch.stack(
        [torch.histc(t , min=0, max=1, bins=11) for t in t]
        ).mean(dim=0)
    histograms_std = torch.stack(
        [torch.histc(t , min=0, max=1, bins=11) for t in t]
        ).std(dim=0)

    mean = t.mean(dim=0)
    std = t.std(dim=0)

    return dict(
        histograms_mean=histograms_mean,
        histograms_std=histograms_std,
        mean=mean,
        std=std,
    )

def _get_activations_metrics(t):
    # t es el tensor con activaciones NO AGREGADAS

    sparsity = (t == 0).sum(dim=-1).float() / t.size(-1)

    return dict(
        sparsity=sparsity,
    )


def get_all_metrics(results_per_qtype,
    identity_names=identity_keys,
    mha_names=mha_keys,
    ):

    qtypes = list(results_per_qtype.keys())

    metrics_per_qtype = {
        qtype: {
            **{identity_name: _get_activations_metrics(
                results_per_qtype[qtype][identity_name]
                ) for identity_name in identity_names},
            **{mha_name: _get_mask_metrics(
                results_per_qtype[qtype][mha_name]
                ) for mha_name in mha_names},
        } for qtype in qtypes
    }

    agg_metrics = {
        **{identity_name: _get_activations_metrics(
            torch.cat([results_per_qtype[qtype][identity_name] for qtype in qtypes])
            ) for identity_name in identity_names},
        **{mha_name: _get_mask_metrics(
            torch.cat([results_per_qtype[qtype][mha_name] for qtype in qtypes])
            ) for mha_name in mha_names},
    }

    return dict(
        metrics_per_qtype=metrics_per_qtype,
        agg_metrics=agg_metrics
    )

def get_val_dataset(clevr_dir, dictionaries):
    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
    return ClevrDataset(
        clevr_dir,
        False,
        dictionaries,
        test_transforms,
    ) 

def gen_selected_idxs_per_qtype(idxs_per_qtype, n=2000):
    return {
        qtype: np.sort(
            np.random.choice(idxs, size=n, replace=False)).tolist()
        for qtype, idxs in idxs_per_qtype.items()
    }

def init_selected_datasets(
        clevr_dir,
        dictionaries_fp,
        selected_idxs_per_qtype_fp,
    ):
    with open(dictionaries_fp, 'rb') as f:
        dictionaries = pickle.load(f)
    
    ds = get_val_dataset(clevr_dir, dictionaries)
    with open(selected_idxs_per_qtype_fp, 'r') as f:
        selected_idxs_per_qtype = json.load(f)
    
    return get_selected_ds_per_qtype(ds, selected_idxs_per_qtype), dictionaries

def load_model(model, weights_fp, data_parallel=True, device_ids=None):
    if data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(weights_fp, map_location='cpu'))
    if data_parallel:
        model = model.module
    return model

def plot_masks_per_qtype(results, mha_keys=mha_keys):
    fig, axes = plt.subplots(ncols=1, nrows=len(mha_keys), figsize=(12, 16), sharex=False, constrained_layout=True)
    qtypes = results.keys()

    # norm = colors.LogNorm()
    for i, k in enumerate(mha_keys):
        ax = axes[i]
        img = ax.imshow(
            np.array([results[qtype][k].mean(dim=0).numpy() for qtype in qtypes]),
            aspect='auto',
            cmap='Greys_r',
        )
        ax.set_title(k)
        ax.set_yticks(list(range(len(qtypes))))
        ax.set_yticklabels(qtypes)
        ax.set_xticks([64 * i for i in range(img.get_size()[1] // 64)])
        cbar = fig.colorbar(img, ax=ax)
        
    plt.show()

if __name__ == '__main__':

    selected_ds_per_qtype, dictionaries = init_selected_datasets(
        '/Users/sebamenabar/Documents/datasets/CLEVR/CLEVR_v1.0/',
        'viz/CLEVR_built_dictionaries.pkl',
        'viz/selected_idxs_per_qtype.json',
    )
    
    with open('config.json', 'r') as f:
        hyp = json.load(f)['hyperparams']['original-fp']

    args = AttrDict()
    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])



