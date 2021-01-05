from pathlib import Path


class DatasetConfiguration:

    def __init__(self,
                 dataset_name,
                 train_images_path=None,
                 train_labels_path=None,
                 train_masks_path=None,
                 test_images_path=None,
                 test_labels_path=None,
                 test_masks_path=None,
                 validation_images_path=None,
                 validation_labels_path=None,
                 image_width=None,
                 image_height=None):
        self.dataset_name = dataset_name
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.train_masks_path = train_masks_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path
        self.test_masks_path = test_masks_path
        self.val_images_path = validation_images_path
        self.val_labels_path = validation_labels_path
        self.image_width = image_width
        self.image_height = image_height

    @staticmethod
    def get_datasets_configuration(root, mode=''):
        dataset_configs = []
        root = Path(root)
        if mode == 'test':
            for dataset in ["DRIVE", "STARE", "CHASE"]:
                troot = root / dataset
                dataset_config = DatasetConfiguration(
                    dataset_name=dataset,
                    test_images_path=str(troot / "images"),
                    test_masks_path=str(troot / "masks"),
                )
                dataset_configs.append(dataset_config)
        else:
            for dataset in ["DRIVE", "STARE", "CHASE"]:
                troot = root / dataset
                dataset_config = DatasetConfiguration(
                    dataset_name=dataset,
                    train_images_path=str(troot / "train/images"),
                    train_labels_path=str(troot / "train/labels")
                )
                dataset_configs.append(dataset_config)

        return dataset_configs