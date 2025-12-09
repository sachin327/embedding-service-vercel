from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from service import get_embedding_service

app = FastAPI()


class EmbeddingRequest(BaseModel):
    inputs: List[str]


@app.post("/embedding")
async def get_embeddings(request: EmbeddingRequest):
    try:
        # The embedding.embed method expects a list of strings and returns a list of list of floats
        embd_srvice = get_embedding_service()
        results = embd_srvice.embed(request.inputs)

        # Ensure results are serializable (list of lists)
        if hasattr(results, "tolist"):
            results = results.tolist()

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    embd_srvice = get_embedding_service()
    return {
        "model_name": embd_srvice.model_name,
        "embedding_dimension": embd_srvice.get_dimension(),
    }
