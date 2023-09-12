import json
import typing as tp
import fire

from GLM.glm_vectorizer import GLMVectorizer
from GLM.tests.test_glm import _construct_samples


def _get_samples_ids(samples_path: str) -> tp.List[str]:
    with open(samples_path) as f:
        samples = json.load(f)
    return [sample['id'] for sample in samples]


def _store_vectors(vectors: tp.Iterable, samples_ids: tp.Iterable, output_path: str) -> None:
    samples = [{'sample_id': sample_id, 'context_output': vector} for sample_id, vector in zip(samples_ids, vectors)]
    with open(output_path, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


def run_glm(encoder_name: str, samples_path: str, weights_path: str, output_path: str, device: str = 'cuda') -> None:
    vectorizer = GLMVectorizer(encoder_name=encoder_name, encoder_weights_path=weights_path, device=device)
    samples = _construct_samples(samples_path)
    samples_ids = _get_samples_ids(samples_path)
    vectors = vectorizer.predict(samples)
    _store_vectors(vectors, samples_ids, output_path)


if __name__ == '__main__':
    fire.Fire(run_glm)
