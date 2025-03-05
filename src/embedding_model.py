import gc
import torch
from transformers import AutoModel, AutoTokenizer

from logging_config import get_logger

logger = get_logger(__name__)


def clear_cuda_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _try_load_model(model_name_or_path: str):
    logger.info(f"Loading model: {model_name_or_path}")
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_safetensors=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model


def _try_load_tokenizer(model_name_or_path: str, max_length: int):
    logger.info(f"Loading tokenizer: {model_name_or_path}")
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        model_max_length=max_length,
        return_tensors="pt",
    )


class EmbeddingModel:
    MODEL_ID = "codesage/codesage-large-v2"
    MODEL_CONTEXT_LENGTH = 1024

    def __init__(self):
        clear_cuda_caches()

        self.model = _try_load_model(self.MODEL_ID)
        self.tokenizer = _try_load_tokenizer(self.MODEL_ID, self.MODEL_CONTEXT_LENGTH)
        self.device = next(self.model.parameters()).device

    def generate(self, text: list[str]):
        logger.debug(f"Generate: {text}")

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
        return embedding.tolist()
