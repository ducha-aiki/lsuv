from __future__ import print_function
import torch
import torch.nn.init
import torch.nn as nn
import re
from typing import List, Optional

gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['scale_to_apply'] = 1.0


def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    return obj


def store_activations(self, input, output):
    gg['act_dict'] = output.detach().cpu()
    return


def is_relevant_layer(m, ignore_layers_regexp_list=None):
    relevant_layers = [nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    valid =  any([isinstance(m, l) for l in relevant_layers])
    if valid and (ignore_layers_regexp_list is not None):
        for r in ignore_layers_regexp_list:
            match = re.search(r, str(m)) 
            if match:
                valid = False
                break
    return valid


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if is_relevant_layer(m, gg['ignore_layers_regexp']):
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
            print (' hooking layer = ', gg['hook_position'], m)
        else:
            #print m, 'already done, skipping'
            gg['hook_position'] += 1
    return

def count_conv_fc_layers(m):
    if is_relevant_layer(m, gg['ignore_layers_regexp']):
        gg['total_fc_conv_layers'] +=1
    return

def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    return

def orthogonal_weights_init(m):
    if is_relevant_layer(m, gg['ignore_layers_regexp']):
        if hasattr(m, 'weight'):
            torch.nn.init.orthogonal_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return

def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if is_relevant_layer(m, gg['ignore_layers_regexp']):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight'):
                print (f"Applying correction coefficient: {gg['scale_to_apply']}")
                m.weight.data[:] *= float(gg['scale_to_apply'])
                gg['correction_needed'] = False  
            return
    return

def LSUVinit(model,
             data,
             needed_std: float = 1.0,
             std_tol:float  = 0.1,
             max_attempts: int = 10,
             do_orthonorm: bool = True,
             ignore_layers_regexp_list: Optional[List[str]] = None,
             verbose: bool = True,
             device=torch.device('cpu')):
    '''Perform Layer-sequential unit-variance (LSUV) initialization using single batch.
    Args:
        model: torch.nn.model
        data: batch of data to pass through model
        needed_std: target std of activation, default 1.0
        std_tol: tolerance for std, default 0.1
        max_attempts: maximum number of attempts to adjust weights, default 1.0
        do_orthonorm: if True, perform orthonormal initialization
        ignore_layers_regexp_list: list of regexp patterns to ignore
        verbose: if True, print debugging information
        device: torch.device
    Returns:
        model with adjusted weights
    '''

    gg['total_fc_conv_layers']=0
    gg['done_counter']= 0
    gg['hook_position'] = 0
    gg['hook']  = None
    gg['ignore_layers_regexp'] = ignore_layers_regexp_list
    model.eval()
    data_dev = move_to(data, device)
    model = model.to(device)
    if verbose: print( 'Starting LSUV')
    model.apply(count_conv_fc_layers)
    if verbose: print ('Total layers to process:', gg['total_fc_conv_layers'])
    with torch.inference_mode():
        if do_orthonorm:
            model.apply(orthogonal_weights_init)
            if verbose: print ('Orthonorm done')
        for layer_idx in range(gg['total_fc_conv_layers']):
            if verbose: print (layer_idx)
            model.apply(add_current_hook)
            _ = model(data_dev)
            current_std = gg['act_dict'].std()
            current_mean = gg['act_dict'].mean()
            if verbose: print ('std at layer ',layer_idx, ' = ', current_std)
            attempts = 0
            while (abs(current_std - needed_std) > std_tol):
                gg['scale_to_apply'] = needed_std / (current_std  + 1e-8)
                gg['correction_needed'] = True
                model.apply(apply_weights_correction)
                model = model.to(device)
                if verbose: print ('Before the correction, std at layer ',layer_idx, ' = ', current_std, 'mean = ', current_mean)
                _ = model(data_dev)
                current_std = gg['act_dict'].std()
                current_mean = gg['act_dict'].mean()
                if verbose: print ('After the correction, std at layer ',layer_idx, ' = ', current_std, 'mean = ', current_mean)
                attempts+=1
                if attempts > max_attempts:
                    if verbose: print ('Cannot converge in ', max_attempts, 'iterations')
                    break
            if gg['hook'] is not None:
                gg['hook'].remove()
            gg['done_counter']+=1
            gg['counter_to_apply_correction'] = 0
            gg['hook_position'] = 0
            gg['hook']  = None
            if verbose: print ('finish at layer',layer_idx )
        if verbose: print ('LSUV init done!')
    return model

lsuv_with_singlebatch = LSUVinit

def lsuv_with_dataloader(model, dataloader, **kwargs):
    '''Perform Layer-sequential unit-variance (LSUV) initialization using dataloader.
    Args:
        model: torch.nn.model
        dataloader: torch.utils.data.DataLoader
        **kwargs: arguments for LSUVinit
    Returns:
        model with adjusted weights'''
    for batch in dataloader:
        break
    batch = batch[0]
    return LSUVinit(model, batch, **kwargs)


