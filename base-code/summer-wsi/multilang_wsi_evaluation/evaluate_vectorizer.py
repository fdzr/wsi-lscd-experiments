import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from multilang_wsi_evaluation.vectorizer_evaluator import vec_eval


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    """
        --config-path path_to_conf --config-name config
    """
    print(f'Config: {OmegaConf.to_yaml(cfg)}')
    model_vectorizer = instantiate(cfg.model_vec)

    if 'model_vectorizer' in cfg:
        inside_vectorizer = instantiate(cfg.model_vectorizer)
        model_vectorizer.inside_vectorizer = inside_vectorizer

    eval_params = cfg.eval_params
    list_paths_datasets = eval_params.list_paths_datasets
    metrics = eval_params.get('metrics')
    callable_metrics = eval_params.get('callable_metrics')
    verbose = eval_params.get('verbose', True)
    ret_metric = cfg.return_value['metric']
    ret_dataset = cfg.return_value['dataset']
    cfg_clusterer = cfg.model_clusterer

    return vec_eval(
        model_vectorizer,
        cfg_clusterer=cfg_clusterer,
        list_paths_datasets=list_paths_datasets,
        metrics=metrics,
        callable_metrics=callable_metrics,
        opt_metric=ret_metric,
        opt_dataset=ret_dataset,
        verbose=verbose)


if __name__ == '__main__':
    run()
