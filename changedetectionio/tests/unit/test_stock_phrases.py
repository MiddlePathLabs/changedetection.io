import pytest

from changedetectionio.processors.restock_diff.stock_phrases import custom_stock_status


@pytest.mark.parametrize('html,settings,expected', [
    ('<p>¡TEMPORALMENTE    AGOTADO!</p>', {'custom_outofstock_strings': ['temporalmente agotado']}, False),
    ('<p>Disponible <strong>ahora</strong></p>', {'custom_instock_strings': ['disponible ahora']}, True),
    ('<p>unavailable</p>', {'custom_instock_strings': ['available']}, None),
    ('<script>Sold out</script><p>Available now</p>', {'custom_outofstock_strings': ['sold out']}, None),
    ('<style>sold out</style><p>Available</p>', {'custom_outofstock_strings': ['sold out']}, None),
    ('<p>Available now. Sold out.</p>', {'custom_instock_strings': ['available now'], 'custom_outofstock_strings': ['sold out']}, False),
    ('<p>Waiting</p>', {'custom_instock_strings': ['', '  ']}, None),
    ('<p>In voorraad</p>', {'custom_instock_strings': 'Disponible ahora\nIn voorraad'}, True),
    ('<p>Anything</p>', {}, None),
])
def test_stock_phrases(html, settings, expected):
    assert custom_stock_status(html, settings) is expected
