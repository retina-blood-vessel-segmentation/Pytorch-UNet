import mlflow
import os

from pathlib import Path
from config import DatasetConfiguration

project_path = Path('..').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
dataset_root_dir = project_path / 'data'


def train_all(models_root_dir, dataset_root_dir):

    datasets = DatasetConfiguration.get_datasets_configuration(dataset_root_dir, mode='train')

    for dataset_config in datasets[:1]:

        parameters = {
            'train_images_dir': str(project_path / dataset_config.train_images_path),
            'train_labels_dir': str(project_path / dataset_config.train_labels_path),
            'dataset': dataset_config.dataset_name,
            'model_dir': str(models_root_dir / dataset_config.dataset_name / (dataset_config.dataset_name + '-model')),
            'batch-size': 4,
            'learning-rate': 0.0001,
            'epochs': 150,
            'scale': 1
        }

        try:
            mlflow.projects.run(
                uri=str(project_path),
                entry_point='train',
                parameters=parameters,
                experiment_name='UNet',
                use_conda=False
            )
        except mlflow.exceptions.ExecutionException as e:
            print('mlflow run execution failed.')
            print(e)
            pass


if __name__ == '__main__':

    train_all(models_root_dir, dataset_root_dir)