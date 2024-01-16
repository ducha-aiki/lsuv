# Layer-sequential unit-variance (LSUV) initialization for PyTorch

This package for neural network initialization.

## Installation

```
pip install lsuv
```

## Usage



Usage:

    from lsuv import lsuv_with_dataloader, lsuv_with_singlebatch
    ...
    model = lsuv_with_dataloader(model, dataloader, device=torch.device('cpu'))

See examples in [test](test/test_lsuv.py)

LSUV initialization is described in:

Mishkin, D. and Matas, J.,(2015). All you need is a good init. ICLR 2016 [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).


### Previous implementations


Original Caffe implementation  [https://github.com/ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit)

Torch re-implementation [https://github.com/yobibyte/torch-lsuv](https://github.com/yobibyte/torch-lsuv)

PyTorch in fastai [https://github.com/fastai/course-v3/blob/master/nbs/dl2/07a_lsuv.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl2/07a_lsuv.ipynb)

Keras implementation: [https://github.com/ducha-aiki/LSUV-keras](https://github.com/ducha-aiki/LSUV-keras)

Thinc re-implementation [LSUV-thinc](https://github.com/explosion/thinc/blob/e653dd3dfe91f8572e2001c8943dbd9b9401768b/thinc/neural/_lsuv.py)
