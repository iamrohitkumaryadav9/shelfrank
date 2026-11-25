import pytest

from scripts.check_env import low_disk_warning


@pytest.mark.parametrize("free", [8.0e9, 17e9])
def test_enough_disk_is_silent(free):
    assert low_disk_warning(int(free)) is None


def test_low_disk_names_the_amount_and_the_minimum():
    message = low_disk_warning(int(7.94e9))
    assert "7.9 GB" in message and "8 GB" in message
