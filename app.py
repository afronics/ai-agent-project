from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class LeadData(BaseModel):
    name: str
    email: str
    company: str = ""
    message: str
    budget: str = ""

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
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt.message}]
            },
            timeout=30
        )
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-lead")
async def classify_lead(lead: LeadData):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    
    prompt = f"""You are a lead qualification expert. Analyze this lead and respond ONLY with valid JSON, no other text.

Lead Information:
- Name: {lead.name}
- Email: {lead.email}
- Company: {lead.company}
- Message: {lead.message}
- Budget: {lead.budget}

Respond with this exact JSON structure:
{{
  "classification": "hot|warm|cold",
  "score": 1-10,
  "summary": "2 sentence summary of the lead",
  "key_needs": ["need1", "need2"],
  "recommended_action": "specific next step for sales team",
  "urgency": "high|medium|low",
  "auto_reply": "personalized email reply to send to the lead"
}}"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=30
        )
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]
        import json
        parsed = json.loads(ai_response)
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))