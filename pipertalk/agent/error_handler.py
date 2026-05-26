"""Error analysis and recovery helpers for Piper agent tool execution."""

import json
import os
import re
from enum import Enum

import requests

LLM_URL = os.environ.get("LLM_URL", "http://192.168.1.16:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")


class ErrorDecision(Enum):
    RETRY = "retry"
    SKIP = "skip"
    REPLAN = "replan"
    ABORT = "abort"


ERROR_ANALYST_PROMPT = """You are the error recovery module of Pipertalk AI assistant.

A task step has failed. Analyze the error and decide what to do.

DECISIONS:
- retry   : Transient error (network timeout, temporary file lock, race condition).
             The same step can succeed if tried again.
- skip    : This step is not critical and the task can succeed without it.
- replan  : The approach was wrong. A different tool or method should be tried.
- abort   : The task is fundamentally impossible or unsafe to continue.

Also provide:
- A brief explanation of WHY it failed (1 sentence)
- A fix suggestion if decision is replan (what to try instead)
- Max retries: how many times to retry if decision is retry (1 or 2)

Return ONLY valid JSON:
{
  "decision": "retry|skip|replan|abort",
  "reason": "why it failed",
  "fix_suggestion": "what to try instead (for replan)",
  "max_retries": 1,
  "user_message": "Short message to tell the user (max 15 words)"
}
"""


def _ollama_complete(system: str, user: str) -> str:
    try:
        resp = requests.post(
            f"{LLM_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"[ErrorHandler] Ollama call failed: {e}")
        raise


def analyze_error(step: dict, error: str, attempt: int = 1, max_attempts: int = 2) -> dict:
    if attempt >= max_attempts:
        print(f"[ErrorHandler] Max attempts reached for step {step.get('step')} — forcing replan")
        return {
            "decision": ErrorDecision.REPLAN,
            "reason": f"Failed {attempt} times: {error[:100]}",
            "fix_suggestion": "Try a completely different approach or tool",
            "max_retries": 0,
            "user_message": "Trying a different approach.",
        }

    prompt = f"""Failed step:
Tool: {step.get('tool')}
Description: {step.get('description')}
Parameters: {json.dumps(step.get('parameters', {}), indent=2)}
Critical: {step.get('critical', False)}

Error:
{error[:500]}

Attempt number: {attempt}"""

    try:
        text = _ollama_complete(ERROR_ANALYST_PROMPT, prompt)
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        result = json.loads(text)
        decision_str = result.get("decision", "replan").lower()
        decision_map = {
            "retry": ErrorDecision.RETRY,
            "skip": ErrorDecision.SKIP,
            "replan": ErrorDecision.REPLAN,
            "abort": ErrorDecision.ABORT,
        }
        result["decision"] = decision_map.get(decision_str, ErrorDecision.REPLAN)

        if step.get("critical") and result["decision"] == ErrorDecision.SKIP:
            result["decision"] = ErrorDecision.REPLAN
            result["user_message"] = "This step is critical — finding alternative approach."

        print(f"[ErrorHandler] Decision: {result['decision'].value} — {result.get('reason', '')}")
        return result

    except Exception as e:
        print(f"[ErrorHandler] Analysis failed: {e} — defaulting to replan")
        return {
            "decision": ErrorDecision.REPLAN,
            "reason": str(e),
            "fix_suggestion": "Try alternative approach",
            "max_retries": 1,
            "user_message": "Encountered an issue, adjusting approach.",
        }


def generate_fix(step: dict, error: str, fix_suggestion: str) -> dict:
    prompt = f"""A task step failed. Generate a replacement step.

Original step:
Tool: {step.get('tool')}
Description: {step.get('description')}
Parameters: {json.dumps(step.get('parameters', {}), indent=2)}

Error: {error[:300]}
Fix suggestion: {fix_suggestion}

Write a Python script that accomplishes the same goal differently.
Return ONLY the Python code, no explanation."""

    try:
        text = _ollama_complete(
            "You are an expert Python developer. Write clean, complete, working Python code using standard library only.",
            prompt,
        )
        code = re.sub(r"```(?:python)?", "", text).strip().rstrip("`").strip()

        return {
            "step": step.get("step"),
            "tool": "generated_code",
            "description": f"Auto-fix for: {step.get('description')}",
            "parameters": {"description": fix_suggestion, "code": code},
            "depends_on": step.get("depends_on", []),
            "critical": step.get("critical", False),
        }

    except Exception as e:
        print(f"[ErrorHandler] Fix generation failed: {e}")
        return {
            "step": step.get("step"),
            "tool": "generated_code",
            "description": f"Fallback for: {step.get('description')}",
            "parameters": {"description": step.get("description", "")},
            "depends_on": step.get("depends_on", []),
            "critical": step.get("critical", False),
        }
