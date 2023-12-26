# Install in Colab

```
!git clone -b layoutlm3-fix https://github.com/necroshine0/unilm/
!git clone https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet

import os

os.rename("layoutlmv3-base-finetuned-publaynet/model_final.pth", "layoutlmv3-base-finetuned-publaynet/pytorch_model.bin")

!pip3 install -q -r unilm/layoutlmv3/requirements.txt
!pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip3 install -q 'git+https://github.com/facebookresearch/detectron2.git'
!pip3 check detectron2

!cd unilm/layoutlmv3 && pip3 install -e .
!pip3 install -q opencv-python tesseract
```

When importing from the `unilm`, use:
```
import sys
sys.path.append("unilm\\layoutlmv3")

try:
    from unilm.layoutlmv3.layoutlmft.models.layoutlmv3 import LayoutLM3Model
    from unilm.layoutlmv3.examples.object_detection.ditod.config import add_vit_config
except:
    from unilm.layoutlmv3.layoutlmft.models.layoutlmv3 import LayoutLM3Model
    from unilm.layoutlmv3.examples.object_detection.ditod.config import add_vit_config
```
or restart the kernel after lib installation