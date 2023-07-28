import monai
import os
import numpy as np
import pandas as pd
import monai.transforms as transforms
from monai.data.dataset import PersistentDataset, CacheDataset, Dataset
from monai.data import DataLoader


def get_data_dicts(
        ids_path: str,
        shuffle: bool = False,
        conditioned: bool = False,
        conditionings=None,
        max_size = None
):
    """
    Get data dictionaries for label generator training.
    :param ids_path: path to TSV file
    :param shuffle: whether to shuffle the labels
    :param conditioned: if conditioning is required, conditioning columns will be present
    :param conditionings: list of conditioning keywords present on TSV file as columns
    :return:
    """
    df = pd.read_csv(ids_path, sep="\t")
    if shuffle:
        df = df.sample(frac=1, random_state=1)

    data_dicts = []
    for index, row in df.iterrows():
        out_dict = {
            "label": row["label"],
        }

        if conditioned:
            for conditioning in conditionings:
                if conditioning in row.keys():
                    out_dict[conditioning] = float(row[conditioning])

        data_dicts.append(out_dict)

    print(f"Found {len(data_dicts)} subjects.")
    if max_size is not None:
        return data_dicts[:max_size]
    else:
        return data_dicts


def get_training_loaders(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        spatial_size: list,
        conditionings: list = [],
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
        cache_dir=None,
        for_ldm=False,
        max_size = None,
):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """
    # Create cache directory
    if  cache_dir is not None:
        if not os.path.isdir(os.path.join(cache_dir, 'cache')):
            os.makedirs(os.path.join(cache_dir, 'cache'))
        cache_dir = os.path.join(cache_dir, 'cache')

    # Define transformations
    base_transforms = [
        transforms.LoadImaged(keys=['label']),  # Niftis
        transforms.AsChannelFirstd(keys=['label'], channel_dim=-1),
        transforms.CenterSpatialCropd(keys=['label'], roi_size=spatial_size),
        transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='symmetric'),
        transforms.Lambdad(keys=['label'], func = lambda l: np.concatenate(
            [np.expand_dims(1-np.sum(l[1:, ...], 0),0), l[1:,...]], 0)),
        transforms.ToTensord(keys=["label", ] + conditionings)
    ]

    val_transforms = transforms.Compose(base_transforms)

    if augmentation:
        if for_ldm:
            rotate_range = [-0.05, 0.05]
            shear_range = [0.001, 0.05]
            scale_range = [0, 0.05]
        else:
            rotate_range = [-0.1, 0.1]
            shear_range = [0.001, 0.15],
            scale_range = [0, 0.3]

        train_transforms = transforms.Compose(
            base_transforms[:-1] + \
            [
                transforms.RandAffined(
                keys=["label"],
                prob=0.0,
                rotate_range=rotate_range,
                shear_range=shear_range,
                scale_range=scale_range,
                padding_mode='border',
                mode='nearest',

            ),
            ] +
            [base_transforms[-1]]
        )

    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings,

    )

    if cache_dir is not None:
        val_ds = PersistentDataset(
            cache_dir = cache_dir,
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        ids_path=training_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings,
        max_size=max_size
    )
    if cache_dir is not None:
        train_ds = PersistentDataset(
            cache_dir=cache_dir,
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return train_loader, val_loader


def get_testing_loader(
        batch_size: int,
        testing_ids: str,
        spatial_size: list,
        conditionings: list = [],
        drop_last: bool = False,
        num_workers: int = 8,
        cache_dir=None,

):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """

    # Create cache directory
    if cache_dir is not None:
        if not os.path.isdir(os.path.join(cache_dir, 'cache')):
            os.makedirs(os.path.join(cache_dir, 'cache'))
        cache_dir = os.path.join(cache_dir, 'cache')

    # Define transformations
    test_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['label']),  # Niftis
        transforms.AsChannelFirstd(keys=['label'], channel_dim=-1),
        transforms.CenterSpatialCropd(keys=['label'], roi_size=spatial_size),
        transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='edge'),
        transforms.ToTensord(keys=["label", ] + conditionings)
    ])

    test_dicts = get_data_dicts(
        ids_path=testing_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings
    )

    if cache_dir is not None:
        test_ds = PersistentDataset(
            cache_dir=cache_dir,
            data=test_dicts,
            transform=test_transforms,
        )
    else:
        test_ds = Dataset(
            data=test_dicts,
            transform=test_transforms,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return test_loader



