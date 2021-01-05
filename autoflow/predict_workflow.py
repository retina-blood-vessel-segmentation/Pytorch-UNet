import mlflow
import os

from glob import glob
from pathlib import Path

from config import DatasetConfiguration


project_path = Path('..').resolve()

models_root_dir = project_path / 'models'
dataset_root_dir = project_path / 'data'
results_root_dir = Path('/home/shared/retina/output/UNet')

# mlflow.set_tracking_uri(str(Path('../../mlflow_db_unet').resolve()))


def predict_all():

    dataset_cfgs = DatasetConfiguration.get_datasets_configuration(dataset_root_dir, mode='test')

    for cfg in dataset_cfgs[-1:]:
        print(f'Running an experiment for {cfg.dataset_name} dataset.')
        model_path = str(models_root_dir / cfg.dataset_name / (cfg.dataset_name + '-model/unet.pth'))
        output_dir = results_root_dir / f'{cfg.dataset_name}/probability_maps'
        if not output_dir.exists():
            print(f'Output directory does not exist. Creating the directory on path {output_dir}.')
            output_dir.mkdir(parents=True)

        test_images_path = Path(cfg.test_images_path)
        print(f'Loading images from directory {test_images_path}.')

        input_image_paths = glob(str(test_images_path / '*'))
        print(f'Found {len(input_image_paths)} images for prediction.')

        for input_image_path in input_image_paths:
            output_image_path = str(output_dir / (Path(input_image_path).stem + '.png'))

            parameters = {
                'model': model_path,
                'dataset': cfg.dataset_name,
                'input': input_image_path,
                'output': output_image_path,
                'scale': 0.5
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

