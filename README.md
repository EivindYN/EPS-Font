# EPS-Font

Implementation of "Few-shot Font Style Transfer with Extraction of Partial Style"

Setup:
``` bash
  pip install -r requirements.txt
```

Tested on Windows, but should work on Linux as well.

## Usage

- Download [dataset](https://github.com/ligoudaner377/font_translator_gan) from Li et al. 

- Unzip it to .\datasets\

- Start visdom (training visualization):
``` bash
  python -m visdom.server
```

- To train:
``` bash
  python train.py
```

NB: Defaults are batch_size = 64, gpu = 0, lambda_SC = 0.

- To test:
``` bash
  python test.py
```
