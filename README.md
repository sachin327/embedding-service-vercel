# Embedding Service

This is a FastAPI-based embedding service that generates text embeddings using `FastEmbed` (or `SentenceTransformers` as a fallback).

## Deployment on Vercel

This project is configured for deployment on [Vercel](https://vercel.com/).

### Prerequisites

- A Vercel account.
- [Vercel CLI](https://vercel.com/docs/cli) installed (optional, for deploying from CLI).

### Configuration

The service uses the following environment variables:

- `EMBEDDING_MODEL_NAME`: (Optional) The name of the embedding model to use. Defaults to `all-MiniLM-L6-v2`.

### Deploying

1.  **Push to GitHub/GitLab/Bitbucket**:
    - Push this repository to your git provider.
    - Import the project in Vercel.
    - Vercel should automatically detect the configuration in `vercel.json` and `requirements.txt`.

2.  **Using Vercel CLI**:
    ```bash
    vercel
    ```

### API Endpoints

-   `POST /embedding`: Generate embeddings for a list of texts.
    -   Body: `{"inputs": ["text1", "text2"]}`
    -   Response: `[[0.1, ...], [0.2, ...]]`
-   `GET /info`: Get information about the loaded model and embedding dimension.

## Local Development

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the server:
    ```bash
    uvicorn main:app --reload
    ```
