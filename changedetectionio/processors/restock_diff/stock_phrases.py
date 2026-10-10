"""Literal stock phrases matched against visible page text."""

import re
import unicodedata


def normalize_stock_text(text):
    text = unicodedata.normalize('NFD', text.casefold())
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return re.sub(r'\s+', ' ', text).strip()


def custom_stock_status(html_content, settings):
    """Return True/False for a custom match, otherwise None for automatic detection.

    Ignore script, style and metadata content, and require phrase boundaries so
    'available' does not match 'unavailable'. Out-of-stock phrases take precedence.
    """
    from bs4 import BeautifulSoup

    unavailable = settings.get('custom_outofstock_strings') or []
    available = settings.get('custom_instock_strings') or []
    if not unavailable and not available:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'template', 'noscript', 'head']):
        element.decompose()
    text = normalize_stock_text(soup.get_text(' ', strip=True))
    for phrases, status in ((unavailable, False), (available, True)):
        if isinstance(phrases, str):
            phrases = phrases.splitlines()
        for phrase in phrases:
            normalized = normalize_stock_text(phrase)
            if normalized and re.search(r'(?<!\w)' + re.escape(normalized) + r'(?!\w)', text):
                return status
    return None
