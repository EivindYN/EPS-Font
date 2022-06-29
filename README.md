# EPS-Font DTS

Implementation of "Few-shot Font Style Transfer with Extraction of Partial Style"

Tested on Windows, but should work on Linux as well.

Setup:
``` bash
  pip install -r requirements.txt
  pip install scikit-image
```

## Usage

- Download [dataset](https://github.com/ligoudaner377/font_translator_gan) from Li et al. 

- Unzip it to .\datasets\

- Make DTS dataset (Warning: Takes a lot of time)
``` bash
  python DTS_dataset.py
```

- Start visdom (training visualization):
``` bash
  python -m visdom.server
```

- To train P(d_{offset}|c_{sk},s):
``` bash
  python train.py --stage 0
```

- To train P(y|d,s):
``` bash
  python train.py --stage 1
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
