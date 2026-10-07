import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import onnxruntime
import torch
from presidio_analyzer import AnalysisExplanation, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.chunkers import CharacterBasedTextChunker
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

from app.document_validators import is_valid_cnpj, is_valid_cpf

LANGUAGE = "pt"
GLINER_MODEL_NAME = "urchade/gliner_multi_pii-v1"
GLINER_THRESHOLD = 0.3
GLINER_THREADS = int(os.environ.get("GLINER_THREADS", "10"))
GLINER_MAX_BATCH_SIZE = 32
GLINER_CHUNK_SIZE = 600
GLINER_CHUNK_OVERLAP = 60
GLINER_ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "gliner_multi_pii-v1-onnx"
GLINER_ONNX_FILE = "model.onnx"

GLINER_ENTITY_MAPPING = {
    "person": "PERSON",
    "phone number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "address": "ADDRESS",
    "date of birth": "DATE_OF_BIRTH",
}

MONTH_NAMES = "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
YEAR_RANGE_AHEAD = r"(?!(?:19|20)\d{2}[\s-](?:19|20)\d{2}(?!\d))"
NUMBER_START = r"(?<!\d)(?<!\d[.\-/])"
NUMBER_END = r"(?!\d)(?![.\-/]\d)"
ADDRESS_COMPLEMENT = r"(?:\s*[,-]?\s*(?:apto?\.?|apartamento|casa|bloco|bl\.?|sala|conj\.?|cj\.?|lote|qd\.?|quadra|andar)\s*[\w-]+)*"
NEIGHBORHOOD = r"(?:\s+[-–]\s+[^\n,;|]{2,40}?(?=\s*,))?"
EMPLOYMENT_PERIOD = re.compile(r"^\d{1,2}/\d{4}\s*[–—-]\s*(?:\d{1,2}/\d{4}|atual|presente|hoje)$", re.IGNORECASE)
STREET_TYPES = r"Rua|R\.|Avenida|Av\.?|Alameda|Al\.|Travessa|Tv\.|Praça|Pça\.?|Estrada|Estr\.|Rodovia|Rod\.|Largo|Viela|Servidão"
BRAZILIAN_STATES = "AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO"
CITY_AND_STATE_SUFFIX = re.compile(
    rf"(?:^|[\s,–-]+)[^\W\d][^\n,;|/\d]{{0,40}}?\s*(?:/|\s[-–]\s|,)\s*(?:{BRAZILIAN_STATES})\.?\s*$"
)
STREET_ADDRESS_HINT = re.compile(rf"\d|\b(?:{STREET_TYPES})", re.IGNORECASE)


class CpfRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="BR_CPF",
            supported_language=LANGUAGE,
            patterns=[
                Pattern("cpf", rf"{NUMBER_START}\d{{3}}\.?\d{{3}}\.?\d{{3}}-?\d{{2}}{NUMBER_END}", 0.5),
            ],
        )

    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_cpf(pattern_text)


class CnpjRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="BR_CNPJ",
            supported_language=LANGUAGE,
            patterns=[
                Pattern("cnpj", rf"{NUMBER_START}\d{{2}}\.?\d{{3}}\.?\d{{3}}/?\d{{4}}-?\d{{2}}{NUMBER_END}", 0.5),
            ],
        )

    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_cnpj(pattern_text)


class BrazilianPhoneRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="PHONE_NUMBER",
            supported_language=LANGUAGE,
            patterns=[
                Pattern(
                    "phone_with_area_code",
                    r"(?<![\w+])(?:\+\s?55[\s-]?|55[\s-])?(?:\(0?[1-9]{2}\)|0?[1-9]{2})[\s-]?(?:9[\s-]?\d{4}|[2-5]\d{3})[\s-]?\d{4}(?![\w])",
                    0.7,
                ),
                Pattern(
                    "phone_without_area_code",
                    rf"(?<!\w)(?<!\d[./-]){YEAR_RANGE_AHEAD}(?:9\s?\d{{4}}|[2-5]\d{{3}})[\s-]\d{{4}}(?!\w)(?![./-]\d)",
                    0.5,
                ),
                Pattern("unformatted_mobile_without_area_code", r"(?<!\w)(?<!\d[./-])9\d{8}(?!\w)(?![./-]\d)", 0.4),
            ],
        )


class CepRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="BR_CEP",
            supported_language=LANGUAGE,
            patterns=[
                Pattern("cep_formatted", rf"{NUMBER_START}\d{{2}}\.?\d{{3}}-\d{{3}}{NUMBER_END}", 0.6),
                Pattern("cep_labeled", r"(?:(?<=CEP)|(?<=CEP:)|(?<=CEP )|(?<=CEP: ))\d{8}(?!\d)", 0.6),
            ],
        )


class EmailAddressRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            supported_language=LANGUAGE,
            patterns=[
                Pattern("email", r"[\w.+%-]+@[\w-]+(?:\.[\w-]+)+", 0.9),
            ],
        )


class SocialProfileUrlRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="SOCIAL_PROFILE_URL",
            supported_language=LANGUAGE,
            patterns=[
                Pattern(
                    "social_profile_url",
                    r"(?:https?://)?(?:[\w-]+\.)?(?:linkedin\.com/(?:in|pub)|github\.com|gitlab\.com|instagram\.com|twitter\.com|x\.com|facebook\.com|behance\.net|medium\.com)/[\w\-./%@]+",
                    0.9,
                ),
            ],
        )


class DateOfBirthRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="DATE_OF_BIRTH",
            supported_language=LANGUAGE,
            patterns=[
                Pattern(
                    "numeric_full_date",
                    rf"{NUMBER_START}(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{{2}}{NUMBER_END}",
                    0.5,
                ),
                Pattern(
                    "written_full_date",
                    rf"\b(?:0?[1-9]|[12]\d|3[01])\s+de\s+(?:{MONTH_NAMES})\s+de\s+(?:19|20)\d{{2}}\b",
                    0.5,
                ),
            ],
        )


class StreetAddressRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="ADDRESS",
            supported_language=LANGUAGE,
            patterns=[
                Pattern(
                    "street_with_number",
                    rf"\b(?:{STREET_TYPES})\s+[^\n,;|]{{2,60}}?,?\s*(?:n[º°o]\.?\s*)?\d{{1,5}}[A-Za-z]?\b{ADDRESS_COMPLEMENT}{NEIGHBORHOOD}",
                    0.6,
                ),
            ],
        )


class PortugueseGlinerRecognizer(GLiNERRecognizer):
    def __init__(self):
        super().__init__(
            supported_language=LANGUAGE,
            entity_mapping=GLINER_ENTITY_MAPPING,
            threshold=GLINER_THRESHOLD,
            map_location="cpu",
            text_chunker=CharacterBasedTextChunker(chunk_size=GLINER_CHUNK_SIZE, chunk_overlap=GLINER_CHUNK_OVERLAP),
            **_gliner_model_options(),
        )

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        chunks = self.text_chunker.chunk(text)
        if not chunks:
            return []
        predictions = self.gliner.inference(
            texts=[chunk.text for chunk in chunks],
            labels=self.gliner_labels,
            flat_ner=self.flat_ner,
            threshold=self.threshold,
            multi_label=self.multi_label,
            batch_size=min(len(chunks), GLINER_MAX_BATCH_SIZE),
        )
        results = [
            self._to_result(prediction, chunk.start)
            for chunk, chunk_predictions in zip(chunks, predictions)
            for prediction in chunk_predictions
        ]
        requested = [
            result for result in results if result.entity_type in entities and not _is_employment_period(text, result)
        ]
        return self.text_chunker.deduplicate_overlapping_entities(requested)

    def _to_result(self, prediction: Dict, offset: int) -> RecognizerResult:
        entity_type = self.model_to_presidio_entity_mapping.get(prediction["label"], prediction["label"])
        return RecognizerResult(
            entity_type=entity_type,
            start=prediction["start"] + offset,
            end=prediction["end"] + offset,
            score=prediction["score"],
            analysis_explanation=AnalysisExplanation(
                recognizer=self.name,
                original_score=prediction["score"],
                textual_explanation=f"Identified as {entity_type} by GLiNER",
            ),
        )


def _is_employment_period(text: str, result: RecognizerResult) -> bool:
    return result.entity_type == "DATE_OF_BIRTH" and bool(EMPLOYMENT_PERIOD.match(text[result.start:result.end].strip()))


def address_without_city_and_state(surface: str) -> str:
    suffix = CITY_AND_STATE_SUFFIX.search(surface)
    street = surface[:suffix.start()] if suffix else surface
    return street if STREET_ADDRESS_HINT.search(street) else ""


def gliner_runtime() -> str:
    return "onnx" if (GLINER_ONNX_DIR / GLINER_ONNX_FILE).is_file() else "torch"


def _gliner_model_options() -> dict:
    if gliner_runtime() == "torch":
        return {"model_name": GLINER_MODEL_NAME}
    return {
        "model_name": str(GLINER_ONNX_DIR),
        "load_onnx_model": True,
        "onnx_model_file": GLINER_ONNX_FILE,
        "load_tokenizer": True,
        "local_files_only": True,
        "runtime_options": {"session_options": _onnx_session_options()},
    }


def _onnx_session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = GLINER_THREADS
    return options


def configure_torch_threads() -> None:
    torch.set_num_threads(GLINER_THREADS)


def build_pattern_recognizers() -> list:
    return [
        CpfRecognizer(),
        CnpjRecognizer(),
        BrazilianPhoneRecognizer(),
        CepRecognizer(),
        SocialProfileUrlRecognizer(),
        DateOfBirthRecognizer(),
        StreetAddressRecognizer(),
        EmailAddressRecognizer(),
    ]
