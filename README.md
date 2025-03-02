# eva-code-agent

https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R/

## Installation
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/py312 --python 3.12 --seed
source ~/py312/bin/activate

uv pip install -r requirements.txt
```

```bash
git clone https://huggingface.co/codesage/codesage-large-v2
git clone https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R
```


## Running
``` bash
#export VLLM_USE_V1=1
vllm serve codesage/codesage-large-v2 --trust-remote-code --task embed
vllm serve Salesforce/SFR-Embedding-Code-2B_R --trust-remote-code --task embed --tensor-parallel-size 1 --disable-sliding-window --enable-chunked-prefill false
```
