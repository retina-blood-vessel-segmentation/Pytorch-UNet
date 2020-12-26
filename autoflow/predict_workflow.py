import mlflow
import os

from glob import glob
from pathlib import Path

from config import DatasetConfiguration


project_path = Path('..').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
dataset_root_dir = project_path / 'data'


def predict_all():

    dataset_cfgs = DatasetConfiguration.get_datasets_configuration(dataset_root_dir, mode='test')

    for cfg in dataset_cfgs[:1]: # just drive for now
        model_path = str(models_root_dir / cfg.dataset_name / (cfg.dataset_name + '-model/unet.pth'))
        output_dir = results_root_dir / cfg.dataset_name / (cfg.dataset_name + '-model')
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        test_images_path = Path(cfg.test_images_path)
        input_image_paths = glob(str(test_images_path / '*'))
        for input_image_path in input_image_paths:
            output_image_path = str(output_dir / (Path(input_image_path).stem + '.png'))

            parameters = {
                'model': model_path,
                'input': input_image_path,
                'output': output_image_path,
                'mask-threshold': 0.5,
                'scale': 1
            }

            mlflow.projects.run(
                uri=str(project_path),
                entry_point='predict',
                parameters=parameters,
                experiment_name='UNet',
                use_conda=False
            )


if __name__ == '__main__':
    predict_all()

