import torch
from transformers import AutoModel, AutoTokenizer

from logging_config import get_logger

logger = get_logger(__name__)


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
    MODEL_CONTEXT_LENGTH = 4096

    def __init__(self):
        self.model = _try_load_model(self.MODEL_ID)
        self.tokenizer = _try_load_tokenizer(self.MODEL_ID, self.MODEL_CONTEXT_LENGTH)
        self.device = next(self.model.parameters()).device

    def generate(self, text: str):
        logger.info(f"Generate: {text}")
        inputs = self.tokenizer.encode(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(inputs)

        embeddings = model_output[0]
        logger.info(f"Generate: {model_output[0].shape}")
        return embeddings.tolist()
