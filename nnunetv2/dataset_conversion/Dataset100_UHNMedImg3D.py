import os
import shutil
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded dataset dir. Should contain 'train', 'validation' and 'test' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=100, help="nnU-Net Dataset ID, default: 100"
    )
    args = parser.parse_args()
    print("Converting...")

    dataset_name = f"Dataset{args.dataset_id:03d}_{'UHNMedImg3D'}"
    train_keys = []
    for i in [0, 1, 2]:
        train_dir = join(args.input_folder, 'train', f'subtype{i}')
        for file in os.listdir(train_dir):
            if file.endswith('_0000.nii.gz'):
                shutil.copy(join(train_dir, file), join(nnUNet_raw, dataset_name, 'imagesTr', file))
            elif file.endswith('.nii.gz'):
                shutil.copy(join(train_dir, file), join(nnUNet_raw, dataset_name, 'labelsTr', file))
                train_keys.append(file.split('.')[0])

    validation_keys = []
    for i in [0, 1, 2]:
        validation_dir = join(args.input_folder, 'validation', f'subtype{i}')
        for file in os.listdir(validation_dir):
            if file.endswith('_0000.nii.gz'):
                shutil.copy(join(validation_dir, file), join(nnUNet_raw, dataset_name, 'imagesTr', file))
            elif file.endswith('.nii.gz'):
                shutil.copy(join(validation_dir, file), join(nnUNet_raw, dataset_name, 'labelsTr', file))
                validation_keys.append(file.split('.')[0])

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "pancreas": 1,
            "lesion": 2,
        },
        file_ending=".nii.gz",
        num_training_cases=len(train_keys) + len(validation_keys),
    )

    splits = [{'train': train_keys, 'val': validation_keys} for _ in range(5)]

    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(preprocessed_folder)
    save_json(splits, join(preprocessed_folder, 'splits_final.json'), sort_keys=False)

    # Convert labels to uint8: c3d file.nii.gz -type uint -o file.nii.gz
    # Change pixel spacing of label quiz_0_430.nii.gz and quiz_0_306.nii.gz to match image

    print("Done!")
