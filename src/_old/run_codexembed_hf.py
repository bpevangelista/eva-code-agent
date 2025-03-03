from transformers import AutoModel
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


base_model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Code-2B_R", trust_remote_code=True)


class CustomCodeXEmbedModel2B(base_model.__class__):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_text(self, texts: list[str], batch_size: int = 12, max_length: int = 1024) -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")
        embeddings = []

        # for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.last_token_pool(model_output, encoded_input["attention_mask"])
            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        logging.info(f"Encoded {embeddings.shape[0]} embeddings.")
        logging.info(f"{embeddings}")
        return embeddings.to(dtype=torch.float32).numpy()

    def encode_queries(
        self,
        queries: list[dict],
        batch_size: int = 12,
        max_length: int = 1024,
        **kwargs,
    ) -> np.ndarray:
        task_description = "Given Code or Text, retrieve relevant content."
        all_queries = [get_detailed_instruct(task_description, query) for query in queries]
        return self.encode_text(all_queries, batch_size, max_length)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]],
        batch_size: int = 12,
        max_length: int = 1024,
        **kwargs,
    ) -> np.ndarray:
        all_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
        return self.encode_text(all_texts, batch_size, max_length)


# Instantiate the custom model
custom_model = CustomCodeXEmbedModel2B.from_pretrained("Salesforce/SFR-Embedding-Code-2B_R", trust_remote_code=True)

custom_model.encode_text(
    [
        """

def fibonnaci(n):

"""
    ]
)
quit()

# # Define datasets
# datasets = [
#     "codetrans-dl", "stackoverflow-qa", "apps", "codefeedback-mt",
#     "codefeedback-st", "cosqa", "stackoverflow-qa", "synthetic-text2sql",
#     "codesearchnet", "codesearchnet-ccr"
# ]
#
# for dataset in datasets:
#     tasks = coir.get_tasks(tasks=[dataset])
#     evaluation = COIR(tasks=tasks, batch_size=16)
#     results = evaluation.run(custom_model, output_folder='results')
#     logging.info(f"Results for {dataset}: {results}")
