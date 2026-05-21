from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Models ──────────────────────────────────────────────────────────────────

class LeadData(BaseModel):
    name: str
    email: str
    company: str = ""
    message: str
    budget: str = ""

class Prompt(BaseModel):
    message: str

# ── Groq helper ──────────────────────────────────────────────────────────────

def call_groq(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    res = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AI Agent is running", "version": "2.0", "agents": 5}


@app.post("/ask")
async def ask_ai(prompt: Prompt):
    """
    Multi-agent endpoint used by n8n.
    Runs 5 specialised agents and returns a unified JSON response.
    """
    msg = prompt.message

    try:
        # ── AGENT 1: Classifier ──────────────────────────────────────────
        classification = call_groq(
            system_prompt=(
                "You are a customer support classifier. "
                "Classify the message as 'urgent' (billing issues, account problems, "
                "service outages, double charges, frustrated tone) or 'general' "
                "(questions, feedback, standard requests). "
                "Reply with ONE word only: urgent or general."
            ),
            user_message=msg,
            max_tokens=5
        ).lower().strip()

        if classification not in ("urgent", "general"):
            classification = "general"

        # ── AGENT 2: Sentiment ───────────────────────────────────────────
        sentiment = call_groq(
            system_prompt=(
                "You are a sentiment analyser. "
                "Classify the customer's emotional tone as one of: "
                "'frustrated', 'neutral', or 'happy'. "
                "Reply with ONE word only."
            ),
            user_message=msg,
            max_tokens=5
        ).lower().strip()

        if sentiment not in ("frustrated", "neutral", "happy"):
            sentiment = "neutral"

        # ── AGENT 3: Category ────────────────────────────────────────────
        category = call_groq(
            system_prompt=(
                "You are a support ticket categoriser. "
                "Assign ONE category from: billing, technical, account, shipping, "
                "general. Reply with ONE word only."
            ),
            user_message=msg,
            max_tokens=5
        ).lower().strip()

        valid_categories = ("billing", "technical", "account", "shipping", "general")
        if category not in valid_categories:
            category = "general"

        # ── AGENT 4: Reply Writer ────────────────────────────────────────
        tone = "empathetic and urgent" if sentiment == "frustrated" else "friendly and professional"
        suggested_reply = call_groq(
            system_prompt=(
                f"You are a customer support agent. Write a {tone} reply to the "
                "customer's message. Keep it concise (2-3 sentences). "
                "Start with 'Dear Customer,' and end with 'Best regards, Support Team'. "
                "Do not include subject lines or extra formatting."
            ),
            user_message=msg,
            max_tokens=150
        )

        # ── AGENT 5: Escalation ──────────────────────────────────────────
        escalate = classification == "urgent" and sentiment == "frustrated"

        escalation_reason = ""
        if escalate:
            escalation_reason = call_groq(
                system_prompt=(
                    "You are a support escalation specialist. "
                    "In ONE short sentence, explain why this ticket needs immediate human attention."
                ),
                user_message=msg,
                max_tokens=60
            )

        # ── Unified response ─────────────────────────────────────────────
        return {
            "response": classification,          # backward compat with n8n
            "classification": classification,
            "sentiment": sentiment,
            "category": category,
            "suggested_reply": suggested_reply,
            "escalate": escalate,
            "escalation_reason": escalation_reason,
            "agents_run": 5
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Groq API timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lead")
async def process_lead(lead: LeadData):
    """
    Lead qualification endpoint (kept from original).
    """
    try:
        result = call_groq(
            system_prompt=(
                "You are a lead qualification assistant. "
                "Analyse the lead and return a JSON object with keys: "
                "score (1-10), priority (high/medium/low), reason (one sentence). "
                "Return ONLY valid JSON."
            ),
            user_message=(
                f"Name: {lead.name}\nEmail: {lead.email}\n"
                f"Company: {lead.company}\nMessage: {lead.message}\n"
                f"Budget: {lead.budget}"
            ),
            max_tokens=150
        )

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
