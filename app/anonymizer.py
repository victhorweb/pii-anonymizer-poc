import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NoOpNlpEngine

from app.document_validators import only_digits
from app.recognizers import (
    GLINER_THRESHOLD,
    LANGUAGE,
    PortugueseGlinerRecognizer,
    build_pattern_recognizers,
)

PLACEHOLDER_LABELS = {
    "PERSON": "NOME",
    "PHONE_NUMBER": "TELEFONE",
    "EMAIL_ADDRESS": "EMAIL",
    "BR_CPF": "CPF",
    "BR_CNPJ": "CNPJ",
    "BR_CEP": "CEP",
    "ADDRESS": "ENDERECO",
    "DATE_OF_BIRTH": "DATA_NASCIMENTO",
    "SOCIAL_PROFILE_URL": "LINK",
}
DIGIT_ENTITY_TYPES = {"BR_CPF", "BR_CNPJ", "BR_CEP", "PHONE_NUMBER"}
PLACEHOLDER_PATTERN = re.compile(r"<\s*([A-Z_]+_\d+)\s*>")
SPAN_EDGE_NOISE = " \t\r\n,;:|•·–—"
MIN_PROPAGATION_LENGTH = 3
BRAZIL_COUNTRY_CODE = "55"


@dataclass(frozen=True)
class DetectedSpan:
    start: int
    end: int
    entity_type: str
    score: float
    sources: frozenset

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "DetectedSpan") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class AnonymizationResult:
    masked_text: str
    mapping: Dict[str, str]
    variants: Dict[str, List[str]]
    entities: List[Dict[str, Any]] = field(default_factory=list)


def build_analyzer_engine() -> AnalyzerEngine:
    nlp_engine = NoOpNlpEngine(models=[{"lang_code": LANGUAGE, "model_name": "no_op"}])
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    for recognizer in [*build_pattern_recognizers(), PortugueseGlinerRecognizer()]:
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[LANGUAGE],
        default_score_threshold=GLINER_THRESHOLD,
    )


def normalize_value(entity_type: str, surface: str) -> str:
    if entity_type in DIGIT_ENTITY_TYPES:
        return _normalize_digits(entity_type, surface)
    collapsed = " ".join(surface.split()).casefold()
    if entity_type == "SOCIAL_PROFILE_URL":
        return re.sub(r"^(?:https?://)?(?:www\.)?", "", collapsed).rstrip("/")
    return collapsed


def _normalize_digits(entity_type: str, surface: str) -> str:
    digits = only_digits(surface)
    if entity_type != "PHONE_NUMBER":
        return digits
    if len(digits) in (12, 13) and digits.startswith(BRAZIL_COUNTRY_CODE):
        digits = digits[len(BRAZIL_COUNTRY_CODE):]
    if len(digits) in (11, 12) and digits.startswith("0"):
        digits = digits[1:]
    return digits


class ResumeAnonymizer:
    def __init__(self, analyzer: AnalyzerEngine):
        self.analyzer = analyzer

    def anonymize(self, text: str) -> AnonymizationResult:
        detected = self._detect(text)
        propagated = self._propagate_occurrences(text, detected)
        merged = self._merge_overlapping(detected + propagated)
        return self._replace_with_placeholders(text, merged)

    def _detect(self, text: str) -> List[DetectedSpan]:
        results = self.analyzer.analyze(text=text, language=LANGUAGE, score_threshold=GLINER_THRESHOLD)
        spans = []
        for result in results:
            start, end = _trim_span(text, result.start, result.end)
            if end <= start:
                continue
            source = result.recognition_metadata.get("recognizer_name", "unknown")
            spans.append(DetectedSpan(start, end, result.entity_type, result.score, frozenset({source})))
        return spans

    def _propagate_occurrences(self, text: str, detected: List[DetectedSpan]) -> List[DetectedSpan]:
        propagated = []
        seen_surfaces = set()
        detected_bounds = {(span.start, span.end) for span in detected}
        for span in detected:
            surface = text[span.start:span.end]
            surface_key = (span.entity_type, normalize_value(span.entity_type, surface))
            if len(surface) < MIN_PROPAGATION_LENGTH or surface_key in seen_surfaces:
                continue
            seen_surfaces.add(surface_key)
            for match in _surface_pattern(surface).finditer(text):
                if (match.start(), match.end()) in detected_bounds:
                    continue
                propagated.append(
                    DetectedSpan(match.start(), match.end(), span.entity_type, span.score, frozenset({"Propagation"}))
                )
        return propagated

    def _merge_overlapping(self, spans: List[DetectedSpan]) -> List[DetectedSpan]:
        merged: List[List[DetectedSpan]] = []
        for span in sorted(spans, key=lambda item: (item.start, -item.end)):
            if merged and span.start < max(member.end for member in merged[-1]):
                merged[-1].append(span)
            else:
                merged.append([span])
        return [_combine_cluster(cluster) for cluster in merged]

    def _replace_with_placeholders(self, text: str, spans: List[DetectedSpan]) -> AnonymizationResult:
        placeholder_by_key: Dict[tuple, str] = {}
        counters: Counter = Counter()
        surfaces_by_placeholder: Dict[str, List[str]] = defaultdict(list)
        entities = []
        pieces = []
        cursor = 0
        for span in spans:
            surface = text[span.start:span.end]
            key = (span.entity_type, normalize_value(span.entity_type, surface))
            if key not in placeholder_by_key:
                label = PLACEHOLDER_LABELS.get(span.entity_type, span.entity_type)
                counters[label] += 1
                placeholder_by_key[key] = f"<{label}_{counters[label]}>"
            placeholder = placeholder_by_key[key]
            surfaces_by_placeholder[placeholder].append(surface)
            pieces.append(text[cursor:span.start])
            pieces.append(placeholder)
            cursor = span.end
            entities.append(
                {
                    "start": span.start,
                    "end": span.end,
                    "text": surface,
                    "entity_type": span.entity_type,
                    "placeholder": placeholder,
                    "score": round(span.score, 3),
                    "sources": sorted(span.sources),
                }
            )
        pieces.append(text[cursor:])
        mapping = {placeholder: _canonical_surface(surfaces) for placeholder, surfaces in surfaces_by_placeholder.items()}
        variants = {
            placeholder: surfaces
            for placeholder, surfaces in surfaces_by_placeholder.items()
            if any(surface != mapping[placeholder] for surface in surfaces)
        }
        return AnonymizationResult("".join(pieces), mapping, variants, entities)


def rehydrate(text: str, mapping: Dict[str, str], variants: Optional[Dict[str, List[str]]] = None) -> str:
    occurrences: Counter = Counter()
    exact_variants = variants or {}

    def restore(match: re.Match) -> str:
        placeholder = f"<{match.group(1)}>"
        if placeholder not in mapping:
            return match.group(0)
        index = occurrences[placeholder]
        occurrences[placeholder] += 1
        placeholder_variants = exact_variants.get(placeholder, [])
        if index < len(placeholder_variants):
            return placeholder_variants[index]
        return mapping[placeholder]

    return PLACEHOLDER_PATTERN.sub(restore, text)


def rehydrate_structure(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return rehydrate(value, mapping)
    if isinstance(value, list):
        return [rehydrate_structure(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: rehydrate_structure(item, mapping) for key, item in value.items()}
    return value


def _trim_span(text: str, start: int, end: int) -> tuple:
    while start < end and text[start] in SPAN_EDGE_NOISE:
        start += 1
    while end > start and text[end - 1] in SPAN_EDGE_NOISE:
        end -= 1
    return start, end


def _surface_pattern(surface: str) -> re.Pattern:
    words = [re.escape(word) for word in surface.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


def _combine_cluster(cluster: List[DetectedSpan]) -> DetectedSpan:
    if len(cluster) == 1:
        return cluster[0]
    dominant = max(cluster, key=lambda span: (span.length, span.score))
    start = min(span.start for span in cluster)
    end = max(span.end for span in cluster)
    sources = frozenset().union(*(span.sources for span in cluster))
    return DetectedSpan(start, end, dominant.entity_type, max(span.score for span in cluster), sources)


def _canonical_surface(surfaces: Iterable[str]) -> str:
    ordered = list(surfaces)
    counts = Counter(ordered)
    return max(ordered, key=lambda surface: (counts[surface], -ordered.index(surface)))
