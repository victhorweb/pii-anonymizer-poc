import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import anthropic
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.anonymizer import ResumeAnonymizer, build_analyzer_engine, rehydrate, rehydrate_structure
from app.llm import MODEL, LlmRefusalError, ResumeStructurer
from app.recognizers import GLINER_MODEL_NAME
from app.text_extraction import UnsupportedFileError, extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pii_poc")

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_PAGE = PROJECT_DIR / "app" / "static" / "index.html"
SAMPLE_RESUME = PROJECT_DIR / "samples" / "curriculo_exemplo.txt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    started_at = time.perf_counter()
    logger.info("Carregando GLiNER (%s) em CPU... a primeira execução baixa ~1 GB.", GLINER_MODEL_NAME)
    analyzer = await run_in_threadpool(build_analyzer_engine)
    app.state.anonymizer = ResumeAnonymizer(analyzer)
    app.state.structurer = ResumeStructurer()
    logger.info("GLiNER pronto em %.1fs. Modo LLM: %s", time.perf_counter() - started_at, _llm_mode(app))
    yield


app = FastAPI(title="Anonimizador de currículos (PoC)", lifespan=lifespan)


class TextPayload(BaseModel):
    text: str = Field(min_length=1)


class RehydratePayload(BaseModel):
    masked_text: str
    mapping: Dict[str, str]
    variants: Dict[str, List[str]] = Field(default_factory=dict)


def _llm_mode(application: FastAPI) -> str:
    return "simulated" if application.state.structurer.is_simulated else "anthropic"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_PAGE)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "llm_mode": _llm_mode(request.app), "llm_model": MODEL, "ner_model": GLINER_MODEL_NAME}


@app.get("/sample")
def sample() -> dict:
    return {"text": SAMPLE_RESUME.read_text(encoding="utf-8")}


@app.post("/extract-text")
async def extract_uploaded_text(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        text = extract_text(file.filename, content)
    except UnsupportedFileError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    return {"filename": file.filename, "text": text}


@app.post("/anonymize")
def anonymize(payload: TextPayload, request: Request) -> dict:
    result = request.app.state.anonymizer.anonymize(payload.text)
    return {
        "masked_text": result.masked_text,
        "mapping": result.mapping,
        "variants": result.variants,
        "entities": result.entities,
    }


@app.post("/rehydrate")
def rehydrate_text(payload: RehydratePayload) -> dict:
    return {"text": rehydrate(payload.masked_text, payload.mapping, payload.variants)}


@app.post("/process")
def process(payload: TextPayload, request: Request) -> dict:
    result = request.app.state.anonymizer.anonymize(payload.text)
    name_placeholder = next(
        (entity["placeholder"] for entity in result.entities if entity["entity_type"] == "PERSON"),
        None,
    )
    try:
        outcome = request.app.state.structurer.structure(result.masked_text, name_placeholder)
    except LlmRefusalError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except anthropic.APIStatusError as error:
        raise HTTPException(status_code=502, detail=f"Erro da API Anthropic ({error.status_code}): {error.message}") from error
    except anthropic.APIConnectionError as error:
        raise HTTPException(status_code=502, detail=f"Falha de conexão com a API Anthropic: {error}") from error
    return {
        "simulated": outcome.simulated,
        "model": outcome.model,
        "sent_to_llm": result.masked_text,
        "llm_response": outcome.raw_response,
        "rehydrated": rehydrate_structure(outcome.structured, result.mapping),
        "mapping": result.mapping,
        "entities": result.entities,
    }
