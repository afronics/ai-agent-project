from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class Prompt(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "AI Agent is running"}

@app.post("/ask")
async def ask_ai(prompt: Prompt):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt.message}]
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    except requests.exceptions.HTTPError as e:
    raise HTTPException(status_code=500, detail=f"{str(e)} - {e.response.text}")
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))