import gc
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from logging_config import get_logger
from platform_utils import is_package_available

logger = get_logger(__name__)


def clear_cuda_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _log_torch_devices_info():
    logger.info(
        f"Torch ({torch.__version__}), CPUs: {torch.get_num_threads()}, "
        + f"CUDA_GPUs: {torch.cuda.device_count()}, "
        + f"MPS_GPU: {torch.backends.mps.is_available()}"
    )

    backends = {
        "CPU": torch.backends.cpu.get_cpu_capability(),
        "CU_DNN": torch.backends.cudnn.is_available(),
        "MKL_DNN": torch.backends.mkldnn.is_available(),
        "MKL": torch.backends.mkl.is_available(),
        "OpenMP": torch.backends.openmp.is_available(),
    }
    logger.info(f"  {', '.join(f'{key} {value}' for key, value in backends.items())}")

    for i in range(torch.cuda.device_count()):
        cuda_device = torch.cuda.device(i)
        device_info = torch.cuda.get_device_properties(cuda_device)
        device_memory_in_gb = device_info.total_memory / (1024 * 1024 * 1024)
        flash_attn_2 = is_package_available("flash_attn", "2.4.2")
        logger.info(f"  {i:02d}: {device_info.name} {round(device_memory_in_gb, 2)}GB")
        logger.info(f"      {{bf16: {torch.cuda.is_bf16_supported()}, flash_attn_2: {flash_attn_2}}}")


def _get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda", 0)
    else:
        return torch.device("cpu")


def _try_load_model(model_name_or_path: str, model_device: Any = None):
    logger.info(f"Loading model: {model_name_or_path}")

    # TODO embedding.cpu() fails for bf16
    # use_bf16 = torch.cuda.is_bf16_supported()
    use_bf16 = False
    torch.set_default_dtype(torch.bfloat16 if use_bf16 else torch.float16)

    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        # TODO Implement sdpa support
        # attn_implementation='flash_attention_2' if is_flash_attn_2_available() else 'sdpa',
    )
    if model_device:
        model.to(model_device)
    model.eval()
    return model


def _try_load_tokenizer(model_name_or_path: str, max_length: int):
    logger.info(f"Loading tokenizer: {model_name_or_path}")
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        model_max_length=max_length,
        return_tensors="pt",
    )


class EmbeddingModel:
    BASE_MODEL_ID = "codesage/codesage-base-v2"
    LARGE_MODEL_ID = "codesage/codesage-large-v2"
    MODEL_ID = BASE_MODEL_ID
    MODEL_CONTEXT_LENGTH = 1024

    def __init__(self):
        clear_cuda_caches()
        _log_torch_devices_info()

        self.device = _get_best_device()
        torch.set_default_device(self.device)

        self.model = _try_load_model(self.MODEL_ID, self.device)
        self.tokenizer = _try_load_tokenizer(self.MODEL_ID, self.MODEL_CONTEXT_LENGTH)

    def generate(self, text: list[str]) -> np.ndarray:
        logger.debug(f"Generate: {text}")

        # TODO Break into 1024 chunks, and generate unified mean-pool

        # [B, D_MODEL]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )

        # Get or compute pool output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)

        # Normalize vector
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        embedding = embedding.cpu().numpy()
        # [B, NDARRAY]
        return embedding
