from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from service import get_embedding_service
import asyncio
import httpx
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the keep-alive task
    task = asyncio.create_task(keep_alive())
    yield
    # Shutdown: Cancel the task (optional, but good practice)
    task.cancel()


app = FastAPI(lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    inputs: List[str]


async def keep_alive():
    """Periodically pings the server to keep it active."""
    url = "https://embedding-service-vercel.onrender.com/ping"  # Adjust URL if deployed elsewhere
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await asyncio.sleep(120)  # Ping every 2 minutes
                logger.info(f"Sending keep-alive ping to {url}")
                response = await client.get(url)
                logger.info(f"Keep-alive ping status: {response.status_code}")
            except Exception as e:
                logger.error(f"Keep-alive ping failed: {e}")


@app.get("/ping")
async def ping():
    return {"status": "alive"}


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
