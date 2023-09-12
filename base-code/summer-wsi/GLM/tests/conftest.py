from pathlib import Path

DATA_PATH = Path(__file__).parent.resolve() / 'data'
DEFAULT_SAMPLES_PATH = DATA_PATH / 'rnc_test.json'
DEFAULT_GOLD_VECTORS_PATH = DATA_PATH / 'preds_rnc_test.json'


def pytest_addoption(parser):
    parser.addoption("--weights-path", action="store", default=None)
    parser.addoption("--samples-path", action="store", default=DEFAULT_SAMPLES_PATH)
    parser.addoption("--gold-vectors-path", action="store", default=DEFAULT_GOLD_VECTORS_PATH)
    parser.addoption("--device", action="store", default='cpu')
