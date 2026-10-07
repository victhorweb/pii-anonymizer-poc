import pytest

from app.anonymizer import ResumeAnonymizer, build_detection_engines


@pytest.fixture(scope="session")
def anonymizer() -> ResumeAnonymizer:
    return ResumeAnonymizer(build_detection_engines())
