import requests
import json

response = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    },
    json={
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": "texto de prueba"
    }
)

data = response.json()
embedding = data["data"][0]["embedding"]
print(f"Dimensiones: {len(embedding)}")