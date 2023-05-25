import logging
import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path
from rdkit.Chem import PandasTools


# module info option
logger = logging.getLogger(__name__)


def set_log(path_output, log_message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path_output / log_message),
            logging.StreamHandler()
        ]
    )


def set_cores(cores):
    max_cores = multiprocessing.cpu_count()
    if cores > max_cores:
        logger.info(f"max cpu cores is {max_cores} / Automatically set to max cores")
        n_cores = max_cores
    else:
        n_cores = cores
    return n_cores


def save(data, output, custom='custom'):

    output = output.with_stem(f"{output.stem}_{custom}")
    logger.info(f"Save {custom} data...")

    # save tsv with condition
    if 'vector' in custom or 'log' in custom:
        data.to_csv(output.with_suffix('.tsv'), sep='\t', index=False)
    else:
        # remove ROMol column
        if 'ROMol' in data.columns:
            data.drop(columns='ROMol', inplace=True)

        # save tsv
        data.to_csv(output.with_suffix('.tsv'), sep='\t', index=False)

        # save excel
        data.to_excel(output.with_suffix('.xlsx'), sheet_name='Sheet1', index=False, header=True)

        # save sdf
        cols = [col for col in data.columns]
        PandasTools.AddMoleculeColumnToFrame(data)
        PandasTools.WriteSDF(data, out=str(output.with_suffix('.sdf')), properties=cols)

