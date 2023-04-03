from typing import Any


def abbreviate_snake_case_text(text: str) -> str:
    text = text.strip().strip("_")
    if not text:
        return ""

    return "".join([word[0] for word in text.split("_") if word]).upper()


def get_kepler_tce_label_color(label: str) -> tuple[str, str]:
    match label:
        case "PC":
            return "success", "bg-success-light"
        case "AFP":
            return "warning", "bg-warning-light"
        case "NTP":
            return "danger", "bg-danger-light"
        case _:
            return "primary", "bg-primary-light"


def get_key_for_value(value: Any, dct: dict[Any, Any]) -> Any:
    return [key for key, value_ in dct.items() if value == value_][0]
