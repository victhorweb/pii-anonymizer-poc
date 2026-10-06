import pytest

from app.anonymizer import ResumeAnonymizer, build_analyzer_engine


@pytest.fixture(scope="session")
def anonymizer() -> ResumeAnonymizer:
    return ResumeAnonymizer(build_analyzer_engine())
