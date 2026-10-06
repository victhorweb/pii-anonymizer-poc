import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import anthropic

logger = logging.getLogger("pii_poc.llm")

MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000
SENIORITY_RANGE = (0, 100)

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Placeholder do nome do candidato, exatamente como aparece no texto (ex.: <NOME_1>)."},
        "headline": {"type": "string"},
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "highlights": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company", "role", "start", "end", "highlights"],
                "additionalProperties": False,
            },
        },
        "education": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
        "languages": {"type": "array", "items": {"type": "string"}},
        "seniority_score": {"type": "integer", "description": "Score de senioridade de 0 a 100."},
        "seniority_rationale": {"type": "string"},
    },
    "required": [
        "name",
        "headline",
        "experiences",
        "education",
        "skills",
        "languages",
        "seniority_score",
        "seniority_rationale",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Você é um analista de recrutamento e seleção no Brasil.
Você recebe currículos ANONIMIZADOS: dados pessoais foram trocados por placeholders como <NOME_1>, <TELEFONE_1>, <EMAIL_1>, <CPF_1>, <ENDERECO_1>.
Regras:
- Copie os placeholders exatamente como aparecem, com os sinais < e >. Nunca tente adivinhar, completar ou inventar o dado real por trás deles.
- O campo name deve conter o placeholder do nome do candidato.
- seniority_score é um inteiro de 0 a 100 baseado em tempo de experiência, escopo, liderança e complexidade técnica; explique o critério em seniority_rationale.
- Responda em português."""


class LlmRefusalError(RuntimeError):
    pass


@dataclass
class StructuringOutcome:
    raw_response: str
    structured: Dict[str, Any]
    simulated: bool
    model: str


class ResumeStructurer:
    def __init__(self, api_key: Optional[str] = None):
        resolved_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=resolved_key) if resolved_key else None

    @property
    def is_simulated(self) -> bool:
        return self.client is None

    def structure(self, masked_text: str, name_placeholder: Optional[str]) -> StructuringOutcome:
        if self.is_simulated:
            return self._simulate(masked_text, name_placeholder)
        return self._call_claude(masked_text)

    def _call_claude(self, masked_text: str) -> StructuringOutcome:
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": RESUME_SCHEMA},
            },
            messages=[{"role": "user", "content": f"Estruture este currículo:\n\n{masked_text}"}],
        )
        if response.stop_reason == "refusal":
            raise LlmRefusalError(f"O modelo recusou a requisição: {response.stop_details}")
        raw_response = next(block.text for block in response.content if block.type == "text")
        structured = json.loads(raw_response)
        structured["seniority_score"] = _clamp_seniority(structured.get("seniority_score"))
        logger.info("Claude respondeu (modelo=%s, stop_reason=%s)", response.model, response.stop_reason)
        return StructuringOutcome(raw_response, structured, simulated=False, model=response.model)

    def _simulate(self, masked_text: str, name_placeholder: Optional[str]) -> StructuringOutcome:
        structured = {
            "simulated": True,
            "notice": "ANTHROPIC_API_KEY ausente: resposta simulada ecoando o texto mascarado.",
            "name": name_placeholder or "",
            "headline": "",
            "experiences": [],
            "education": [],
            "skills": [],
            "languages": [],
            "seniority_score": None,
            "seniority_rationale": "",
            "echo": masked_text,
        }
        raw_response = json.dumps(structured, ensure_ascii=False, indent=2)
        return StructuringOutcome(raw_response, structured, simulated=True, model="simulated")


def _clamp_seniority(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)):
        return None
    low, high = SENIORITY_RANGE
    return int(min(max(value, low), high))
