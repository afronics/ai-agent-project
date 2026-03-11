from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class Prompt(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "AI Agent is running"}

@app.post("/ask")
async def ask_ai(prompt: Prompt):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt.message}]
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))