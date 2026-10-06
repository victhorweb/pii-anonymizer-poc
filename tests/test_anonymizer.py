from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import RecognizerResult

from app.anonymizer import ResumeAnonymizer, rehydrate
from app.document_validators import is_valid_cnpj, is_valid_cpf

SAMPLE_RESUME = Path(__file__).resolve().parent.parent / "samples" / "curriculo_exemplo.txt"
VALID_CPF = "529.982.247-25"
INVALID_CPF = "529.982.247-26"


def entities_of_type(result, entity_type):
    return [entity for entity in result.entities if entity["entity_type"] == entity_type]


@pytest.mark.parametrize("cpf", ["529.982.247-25", "52998224725", "111.444.777-35"])
def test_check_digit_accepts_valid_cpf(cpf):
    assert is_valid_cpf(cpf)


@pytest.mark.parametrize("cpf", ["529.982.247-26", "123.456.789-00", "111.111.111-11", "5299822472"])
def test_check_digit_rejects_invalid_cpf(cpf):
    assert not is_valid_cpf(cpf)


def test_check_digit_validates_cnpj():
    assert is_valid_cnpj("11.222.333/0001-81")
    assert not is_valid_cnpj("11.222.333/0001-82")


@pytest.mark.parametrize("cpf", [VALID_CPF, "52998224725"])
def test_valid_cpf_is_detected(anonymizer, cpf):
    result = anonymizer.anonymize(f"Documento do candidato: CPF {cpf}.")

    cpf_entities = entities_of_type(result, "BR_CPF")
    assert [entity["text"] for entity in cpf_entities] == [cpf]
    assert cpf not in result.masked_text
    assert "<CPF_1>" in result.masked_text


@pytest.mark.parametrize("cpf", [INVALID_CPF, "123.456.789-00"])
def test_invalid_cpf_is_ignored(anonymizer, cpf):
    result = anonymizer.anonymize(f"Documento do candidato: CPF {cpf}.")

    assert entities_of_type(result, "BR_CPF") == []
    assert "<CPF_" not in result.masked_text


@pytest.mark.parametrize(
    "phone",
    [
        "(11) 98765-4321",
        "11 98765-4321",
        "11987654321",
        "+55 11 98765-4321",
        "+5511987654321",
        "+55 (21) 3456-7890",
        "(011) 3456-7890",
        "21 3456 7890",
        "98765-4321",
        "3456-7890",
    ],
)
def test_brazilian_phone_formats_are_masked(anonymizer, phone):
    result = anonymizer.anonymize(f"Celular para contato: {phone} (horário comercial)")

    phone_entities = entities_of_type(result, "PHONE_NUMBER")
    assert [entity["text"] for entity in phone_entities] == [phone]
    assert phone not in result.masked_text


def test_numbers_followed_by_sentence_punctuation_are_masked(anonymizer):
    result = anonymizer.anonymize("Ligue 98765-4321. CEP 04117-091. CPF 529.982.247-25; nascida em 14/03/1991.")

    for secret in ["98765-4321", "04117-091", "529.982.247-25", "14/03/1991"]:
        assert secret not in result.masked_text, secret


def test_full_address_with_complement_and_city_is_masked(anonymizer):
    result = anonymizer.anonymize(SAMPLE_RESUME.read_text(encoding="utf-8"))

    for secret in ["apto 31", "Vila Mariana", "São Paulo/SP"]:
        assert secret not in result.masked_text, secret


def test_employment_periods_are_not_mistaken_for_birth_dates(anonymizer):
    result = anonymizer.anonymize(SAMPLE_RESUME.read_text(encoding="utf-8"))

    assert [entity["text"] for entity in entities_of_type(result, "DATE_OF_BIRTH")] == ["14/03/1991"]
    assert "01/2018 – 02/2021" in result.masked_text


def test_year_ranges_are_not_mistaken_for_phones(anonymizer):
    result = anonymizer.anonymize("Analista de Dados na Empresa X (2019-2023) e Empresa Y 2015 – 2017.")

    assert entities_of_type(result, "PHONE_NUMBER") == []


def test_repeated_name_gets_the_same_placeholder(anonymizer):
    text = (
        "Ana Beatriz Ferreira Costa é engenheira de dados.\n"
        "Referências sobre Ana Beatriz Ferreira Costa podem ser solicitadas.\n"
        "Assinado: Ana Beatriz Ferreira Costa"
    )

    result = anonymizer.anonymize(text)

    person_placeholders = {entity["placeholder"] for entity in entities_of_type(result, "PERSON")}
    assert person_placeholders == {"<NOME_1>"}
    assert result.masked_text.count("<NOME_1>") == 3
    assert "Ana Beatriz" not in result.masked_text


class FirstOccurrenceOnlyAnalyzer:
    def __init__(self, surface, entity_type):
        self.surface = surface
        self.entity_type = entity_type

    def analyze(self, text, **_):
        start = text.index(self.surface)
        return [
            RecognizerResult(
                self.entity_type,
                start,
                start + len(self.surface),
                0.9,
                recognition_metadata={"recognizer_name": "Stub"},
            )
        ]


def test_occurrences_missed_by_the_model_inherit_the_placeholder():
    text = "Ana Beatriz Costa, engenheira.\nANA BEATRIZ COSTA\nContato de ana beatriz  costa via RH."
    anonymizer = ResumeAnonymizer(FirstOccurrenceOnlyAnalyzer("Ana Beatriz Costa", "PERSON"))

    result = anonymizer.anonymize(text)

    assert result.masked_text == "<NOME_1>, engenheira.\n<NOME_1>\nContato de <NOME_1> via RH."
    assert result.mapping == {"<NOME_1>": "Ana Beatriz Costa"}
    assert rehydrate(result.masked_text, result.mapping, result.variants) == text


def test_same_value_in_different_formats_shares_placeholder(anonymizer):
    result = anonymizer.anonymize(f"CPF {VALID_CPF}. Declaração assinada, CPF 52998224725.")

    assert {entity["placeholder"] for entity in entities_of_type(result, "BR_CPF")} == {"<CPF_1>"}


def test_detected_spans_never_overlap(anonymizer):
    result = anonymizer.anonymize(SAMPLE_RESUME.read_text(encoding="utf-8"))

    ordered = sorted(result.entities, key=lambda entity: entity["start"])
    for previous, current in zip(ordered, ordered[1:]):
        assert previous["end"] <= current["start"]
    assert "<<" not in result.masked_text
    assert ">>" not in result.masked_text


def test_sample_resume_hides_every_personal_datum(anonymizer):
    text = SAMPLE_RESUME.read_text(encoding="utf-8")

    result = anonymizer.anonymize(text)

    for secret in [
        "Maria Eduarda dos Santos Oliveira",
        "MARIA EDUARDA DOS SANTOS OLIVEIRA",
        "98765-4321",
        "3456-7890",
        "99876 5432",
        "maria.eduarda.oliveira@emailfalso.com.br",
        "linkedin.com/in/maria-eduarda-oliveira-dev",
        "github.com/mariaeduardaoliveira",
        "Rua das Laranjeiras",
        "04117-091",
        "14/03/1991",
        "529.982.247-25",
        "52998224725",
        "11.222.333/0001-81",
    ]:
        assert secret not in result.masked_text, secret


def test_anonymize_then_rehydrate_returns_identical_text(anonymizer):
    text = SAMPLE_RESUME.read_text(encoding="utf-8")

    result = anonymizer.anonymize(text)

    assert result.masked_text != text
    assert rehydrate(result.masked_text, result.mapping, result.variants) == text


def test_rehydrate_ignores_unknown_placeholders():
    assert rehydrate("Olá <NOME_1> e <NOME_9>", {"<NOME_1>": "Ana"}) == "Olá Ana e <NOME_9>"


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("ANTHROPIC_API_KEY", raising=False)
        yield patch


def test_api_roundtrip_through_endpoints(client):
    text = SAMPLE_RESUME.read_text(encoding="utf-8")

    anonymized = client.post("/anonymize", json={"text": text}).json()
    restored = client.post(
        "/rehydrate",
        json={"masked_text": anonymized["masked_text"], "mapping": anonymized["mapping"], "variants": anonymized["variants"]},
    ).json()

    assert restored["text"] == text
    assert {"start", "end", "entity_type", "score", "placeholder"} <= set(anonymized["entities"][0])


def test_process_without_api_key_echoes_masked_text_and_rehydrates(client):
    text = SAMPLE_RESUME.read_text(encoding="utf-8")

    processed = client.post("/process", json={"text": text}).json()

    assert processed["simulated"] is True
    assert "529.982.247-25" not in processed["sent_to_llm"]
    assert "529.982.247-25" not in processed["llm_response"]
    assert "<NOME_1>" in processed["llm_response"]
    assert processed["rehydrated"]["name"] == "Maria Eduarda dos Santos Oliveira"
