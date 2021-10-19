# CoAtNet

## Overview

This is a PyTorch implementation of CoAtNet specified in [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803), arXiv 2021.

![img](https://user-images.githubusercontent.com/67839539/135721248-bcd1b2bc-458b-4be2-a793-7a18cf24f703.png)

👉 Check out [MobileViT](https://github.com/chinhsuanwu/mobilevit-pytorch) if you are interested in other **Convolution + Transformer** models.

## Usage

```python
import torch
from coatnet import coatnet_0

net = coatnet_0()
img = torch.randn(1, 3, 224, 224)
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
