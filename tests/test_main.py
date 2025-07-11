"""Tests for the main module."""

import pytest

from main import main


def test_main_output(capsys):
    """Test that main() prints Hello World!"""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello World!"


def test_main_no_error():
    """Test that main() runs without errors."""
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")
