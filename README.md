# EPS-Font

Implementation of "Few-shot Font Style Transfer with Extraction of Partial Style"

Tested on Windows, but should work on Linux as well.

If you want to use DTS, go to the branch "DTS".

## Setup

``` bash
  pip install -r requirements.txt
```

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

If you get an error ending with:
``` bash
  Pointer addresses:
      input: 000000087C570000
      output: 00000007854CD800
      weight: 0000000785488000
```
Then decrease the batch size.

- To test:
``` bash
  python test.py
```

## Acknowledgement

Code derived and rehashed from:
* [pix2pix](https://github.com/yenchenlin/pix2pix-tensorflow) by Azadi et al.
* [FTransGAN](https://github.com/ligoudaner377/font_translator_gan) by Li et al. 

