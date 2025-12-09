from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from embedding import embedding

app = FastAPI()


class EmbeddingRequest(BaseModel):
    inputs: List[str]


@app.post("/embedding")
async def get_embeddings(request: EmbeddingRequest):
    try:
        # The embedding.embed method expects a list of strings and returns a list of list of floats
        results = embedding.embed(request.inputs)

        # Ensure results are serializable (list of lists)
        if hasattr(results, "tolist"):
            results = results.tolist()

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    return {
        "model_name": embedding.model_name,
        "embedding_dimension": embedding.get_dimension(),
    }
