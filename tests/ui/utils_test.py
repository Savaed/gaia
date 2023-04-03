import pytest

from gaia.ui.utils import abbreviate_snake_case_text, get_kepler_tce_label_color, get_key_for_value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("_", ""),
        ("_lorem_ipsum_dolor", "LID"),
        ("lorem_ipsum_dolor_", "LID"),
        ("_lorem_ipsum_dolor_", "LID"),
        ("__lorem_ipsum_dolor", "LID"),
        ("lorem___ipsum__dolor", "LID"),
        ("LoRem_iPsum_Dolor", "LID"),
        ("LID", "L"),
        ("lId", "L"),
        ("l", "L"),
    ],
)
def test_function__return_correct_abbreviation(text, expected):
    """Test that correct abbreviation is returned."""
    result = abbreviate_snake_case_text(text)
    assert result == expected


@pytest.mark.parametrize(
    "label,expected",
    [
        ("PC", ("success", "bg-success-light")),
        ("AFP", ("warning", "bg-warning-light")),
        ("NTP", ("danger", "bg-danger-light")),
        ("sdfsfsf", ("primary", "bg-primary-light")),
    ],
)
def test_get_kepler_tce_label_color__return_correct_color(label, expected):
    """Test that correct color is returned."""
    result = get_kepler_tce_label_color(label)
    assert result == expected


def test_get_key_for_value__return_correct_key():
    """Test that correct key is returned for a specific value."""
    test_dict = {"a": 1, "b": 2}
    result = get_key_for_value(1, test_dict)
    assert result == "a"
