C:\Users\wladl\Desktop\Dev\2. yeeproject>docker build -t bertje-trainer:latest .
[+] Building 301.3s (11/16)                                                                        docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                               0.0s
 => => transferring dockerfile: 1.51kB                                                                             0.0s
 => [internal] load metadata for docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime                           1.2s
 => [auth] pytorch/pytorch:pull token for registry-1.docker.io                                                     0.0s
 => [internal] load .dockerignore                                                                                  0.0s
 => => transferring context: 468B                                                                                  0.0s
 => [ 1/11] FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime@sha256:e4aaefef0c96318759160ff971b527a  241.9s
 => => resolve docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime@sha256:e4aaefef0c96318759160ff971b527ae61e  0.0s
 => => sha256:002e0840847cf8ebfcf4b678bf422b60efcbff7498b4585ac4297fae8938670e 3.43GB / 3.43GB                   199.6s
 => => sha256:e991862a3134e6341651e737286a9353786cdf1c114f957f97f495cd3001b062 99B / 99B                           0.3s
 => => sha256:f030756bea58ae99a33ff55fa32829b4b114bf6537c43b9a28c5883c6d30ba1e 15.13MB / 15.13MB                   1.6s
 => => sha256:7007490126efaae58924972668256aaeb4858e6c4537eb4257e1978719b958c7 28.58MB / 28.58MB                   2.5s
 => => extracting sha256:7007490126efaae58924972668256aaeb4858e6c4537eb4257e1978719b958c7                          1.4s
 => => extracting sha256:f030756bea58ae99a33ff55fa32829b4b114bf6537c43b9a28c5883c6d30ba1e                          0.8s
 => => extracting sha256:002e0840847cf8ebfcf4b678bf422b60efcbff7498b4585ac4297fae8938670e                         42.1s
 => => extracting sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1                          0.0s
 => => extracting sha256:e991862a3134e6341651e737286a9353786cdf1c114f957f97f495cd3001b062                          0.0s
 => [internal] load build context                                                                                 22.0s
 => => transferring context: 438.49MB                                                                             21.9s
 => [ 2/11] WORKDIR /app                                                                                           0.8s
 => [ 3/11] RUN apt-get update && apt-get install -y --no-install-recommends     git     wget     && rm -rf /var  12.7s
 => [ 4/11] COPY requirements.txt .                                                                                0.1s
 => [ 5/11] RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt    42.1s
 => ERROR [ 6/11] RUN python -c "from transformers import AutoTokenizer, AutoModel;     AutoTokenizer.from_pretra  2.2s
------
 > [ 6/11] RUN python -c "from transformers import AutoTokenizer, AutoModel;     AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased');     AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')":
1.763 Traceback (most recent call last):
1.763   File "<string>", line 1, in <module>
1.763   File "/opt/conda/lib/python3.10/site-packages/transformers/__init__.py", line 27, in <module>
1.763     from . import dependency_versions_check
1.763   File "/opt/conda/lib/python3.10/site-packages/transformers/dependency_versions_check.py", line 16, in <module>
1.763     from .utils.versions import require_version, require_version_core
1.763   File "/opt/conda/lib/python3.10/site-packages/transformers/utils/__init__.py", line 24, in <module>
1.763     from .auto_docstring import (
1.763   File "/opt/conda/lib/python3.10/site-packages/transformers/utils/auto_docstring.py", line 30, in <module>
1.763     from .generic import ModelOutput
1.763   File "/opt/conda/lib/python3.10/site-packages/transformers/utils/generic.py", line 465, in <module>
1.764     _torch_pytree.register_pytree_node(
1.764 AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'. Did you mean: '_register_pytree_node'?
------
Dockerfile:29
--------------------
  28 |     # Pre-download the BERTje model to avoid download during training
  29 | >>> RUN python -c "from transformers import AutoTokenizer, AutoModel; \
  30 | >>>     AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased'); \
  31 | >>>     AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')"
  32 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python -c \"from transformers import AutoTokenizer, AutoModel;     AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased');     AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')\"" did not complete successfully: exit code: 1

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/wlahslo7dgnh4y7nmf45287vx

C:\Users\wladl\Desktop\Dev\2. yeeproject>