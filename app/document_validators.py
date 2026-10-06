import re

NON_DIGITS = re.compile(r"\D")

CPF_FIRST_WEIGHTS = range(10, 1, -1)
CPF_SECOND_WEIGHTS = range(11, 1, -1)
CNPJ_FIRST_WEIGHTS = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
CNPJ_SECOND_WEIGHTS = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)


def only_digits(value: str) -> str:
    return NON_DIGITS.sub("", value)


def _has_repeated_digits(digits: str) -> bool:
    return len(set(digits)) == 1


def _check_digit(digits: str, weights) -> int:
    remainder = sum(int(digit) * weight for digit, weight in zip(digits, weights)) % 11
    return 0 if remainder < 2 else 11 - remainder


def is_valid_cpf(value: str) -> bool:
    digits = only_digits(value)
    if len(digits) != 11 or _has_repeated_digits(digits):
        return False
    first = _check_digit(digits[:9], CPF_FIRST_WEIGHTS)
    second = _check_digit(digits[:10], CPF_SECOND_WEIGHTS)
    return digits[9:] == f"{first}{second}"


def is_valid_cnpj(value: str) -> bool:
    digits = only_digits(value)
    if len(digits) != 14 or _has_repeated_digits(digits):
        return False
    first = _check_digit(digits[:12], CNPJ_FIRST_WEIGHTS)
    second = _check_digit(digits[:13], CNPJ_SECOND_WEIGHTS)
    return digits[12:] == f"{first}{second}"
