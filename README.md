# eva-code-agent

## Installation
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh   
uv venv ~/py312 --python 3.12 --seed   
source ~/py312/bin/activate  

uv pip install -r requirements.txt
```

## Running
``` bash
#export VLLM_USE_V1=1  
vllm serve Salesforce/SFR-Embedding-Code-2B_R --trust-remote-code --task embed --tensor-parallel-size 1 --disable-sliding-window
```