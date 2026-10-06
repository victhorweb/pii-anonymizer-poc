import re
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

from app.document_validators import is_valid_cnpj, is_valid_cpf

LANGUAGE = "pt"
GLINER_MODEL_NAME = "urchade/gliner_multi_pii-v1"
GLINER_THRESHOLD = 0.3

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
CITY_AND_STATE = r"(?:\s*,\s*[^\n,;|/]{2,40}/[a-z]{2}\b)?"
EMPLOYMENT_PERIOD = re.compile(r"^\d{1,2}/\d{4}\s*[–—-]\s*(?:\d{1,2}/\d{4}|atual|presente|hoje)$", re.IGNORECASE)
STREET_TYPES = r"Rua|R\.|Avenida|Av\.?|Alameda|Al\.|Travessa|Tv\.|Praça|Pça\.?|Estrada|Estr\.|Rodovia|Rod\.|Largo|Viela|Servidão"


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
                    rf"\b(?:{STREET_TYPES})\s+[^\n,;|]{{2,60}}?,?\s*(?:n[º°o]\.?\s*)?\d{{1,5}}[A-Za-z]?\b{ADDRESS_COMPLEMENT}{NEIGHBORHOOD}{CITY_AND_STATE}",
                    0.6,
                ),
            ],
        )


class PortugueseGlinerRecognizer(GLiNERRecognizer):
    def __init__(self):
        super().__init__(
            supported_language=LANGUAGE,
            entity_mapping=GLINER_ENTITY_MAPPING,
            model_name=GLINER_MODEL_NAME,
            threshold=GLINER_THRESHOLD,
            map_location="cpu",
        )

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        own_entities = [entity for entity in entities if entity in self.supported_entities]
        results = super().analyze(text, own_entities, nlp_artifacts)
        return [result for result in results if not _is_employment_period(text, result)]


def _is_employment_period(text: str, result: RecognizerResult) -> bool:
    return result.entity_type == "DATE_OF_BIRTH" and bool(EMPLOYMENT_PERIOD.match(text[result.start:result.end].strip()))


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
