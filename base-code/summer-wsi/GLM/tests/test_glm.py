import json
import pytest
import typing as tp

from multilang_wsi_evaluation.utils import Sample
from GLM.glm_vectorizer import GLMVectorizer


@pytest.fixture
def weights_path(pytestconfig):
    return pytestconfig.getoption('weights_path')


@pytest.fixture
def samples_path(pytestconfig):
    return pytestconfig.getoption('samples_path')


@pytest.fixture
def gold_vectors_path(pytestconfig):
    return pytestconfig.getoption('gold_vectors_path')


@pytest.fixture
def device(pytestconfig):
    return pytestconfig.getoption('device')


def _construct_samples(json_samples_path: str) -> tp.List[Sample]:
    with open(json_samples_path) as f:
        samples = json.load(f)
    return [Sample(context=sample['sentence'], begin=sample['start'], end=sample['end'], lemma=sample['lemma'])
            for sample in samples]


def _get_preds_vectors(json_preds_path: str) -> tp.List[tp.List[float]]:
    with open(json_preds_path) as f:
        preds = json.load(f)
    return [pred['context_output'] for pred in preds]


def test_rnc(samples_path: str, gold_vectors_path: str, device: str, weights_path: tp.Optional[str]) -> None:
    assert weights_path is not None
    vectorizer = GLMVectorizer(encoder_name='xlm-roberta-large', encoder_weights_path=weights_path, device=device)
    samples = _construct_samples(samples_path)
    vectors = vectorizer.predict(samples)
    preds_vectors = _get_preds_vectors(gold_vectors_path)
    for vector, pred_vector in zip(vectors, preds_vectors):
        assert vector == pytest.approx(pred_vector, abs=1e-4)
