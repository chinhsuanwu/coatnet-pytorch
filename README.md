# CoAtNet

## Overview

This is a PyTorch implementation of CoAtNet specified in ["CoAtNet: Marrying Convolution and Attention for All Data Sizes"](https://arxiv.org/abs/2106.04803), arXiv 2021.

![img](https://user-images.githubusercontent.com/67839539/138133065-337bb5ac-3dca-4ce8-af51-990c5ff23316.png)

👉 Check out [MobileViT](https://github.com/chinhsuanwu/mobilevit-pytorch) if you are interested in other **Convolution + Transformer** models.

## Usage

```python
import torch
from coatnet import coatnet_0

img = torch.randn(1, 3, 224, 224)
net = coatnet_0()
out = net(img)
```

Try out other block combinations mentioned in the paper:

```python
from coatnet import CoAtNet

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

net = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types)
out = net(img)
```

## Citation

```bibtex
@article{dai2021coatnet,
  title={CoAtNet: Marrying Convolution and Attention for All Data Sizes},
  author={Dai, Zihang and Liu, Hanxiao and Le, Quoc V and Tan, Mingxing},
  journal={arXiv preprint arXiv:2106.04803},
  year={2021}
}
```

## Credits

Code adapted from [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2) and [ViT](https://github.com/lucidrains/vit-pytorch).
