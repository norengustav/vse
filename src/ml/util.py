# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import os

import src.data.util
from src.data.basic_data_parameters import BasicDataParameters


def create_full_model_path(basic_data: BasicDataParameters, basedir: str, epoch: int):
    basic_data_str = src.data.util.basic_data_to_str(basic_data)
    filename = f"model_{basic_data_str}_epoch{epoch}.pt"

    model_fullpath = os.path.join(basedir, filename)
    return model_fullpath
