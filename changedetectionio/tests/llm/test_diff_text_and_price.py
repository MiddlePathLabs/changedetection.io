import pytest

from changedetectionio.llm.diff_text import build_llm_diff
from changedetectionio.processors.restock_diff.plugins.llm_restock import _normalise_price


class TestBuildLlmDiff:
    def test_no_header_and_one_line_per_diff_line(self):
        out = build_llm_diff('a\nb\nc\n', 'a\nB\nc\n')
        lines = out.splitlines()
        assert not any(ln.startswith(('---', '+++')) for ln in lines)
        assert '-b' in lines and '+B' in lines
        assert lines[0].startswith('@@')

    def test_last_line_without_newline_is_not_glued(self):
        """The worker used to produce '-old price+new price' as ONE line here."""
        out = build_llm_diff('item\nPrice $500', 'item\nPrice $400\n')
        lines = out.splitlines()
        assert '-Price $500' in lines
        assert '+Price $400' in lines

    def test_identical_text_gives_empty_diff(self):
        assert build_llm_diff('same\n', 'same') == ''

    def test_ignore_whitespace(self):
        assert build_llm_diff('a  b\n', 'a b\n', ignore_whitespace=True) == ''
        assert build_llm_diff('a  b\n', 'a b\n') != ''


class TestNormalisePrice:
    @pytest.mark.parametrize('raw,expected', [
        ('$1,299.00', 1299.0),
        ('1.299,00 €', 1299.0),
        ('12,50', 12.5),
        ('12.50', 12.5),
        ('1,299', 1299.0),
        ('1.299', 1299.0),
        ('0.999', 0.999),
        ('1 234,56', 1234.56),
        ('1.234.567,89', 1234567.89),
        (19.99, 19.99),
        (5, 5.0),
    ])
    def test_formats(self, raw, expected):
        assert _normalise_price(raw) == pytest.approx(expected)

    def test_no_digits_raises(self):
        with pytest.raises(ValueError):
            _normalise_price('call for price')
