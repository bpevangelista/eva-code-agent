import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from embedding_model import EmbeddingModel

app = FastAPI(title="Embedding Generator API")
embedding_model = EmbeddingModel()


class EmbeddingRequest(BaseModel):
    prompt: str


@app.post("/v1/embeddings")
async def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.generate(request.prompt)
    return {"embedding": embedding}


# --- Run the server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
