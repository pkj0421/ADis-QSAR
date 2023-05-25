import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from Admodule import Reader, Utils

"""
This code is used for data preparation
Need compound data with active values for the target
Input file need 'Smiles' column

For ChEMBL data, a ChEMBL_reader is prepared (use --chembl True option)

< ChEMBL_reader > 
use only IC50, Ki, Kd, %Inhibition values
use only 'single protein format' in BAO Label (in vitro assay)
use only '=' relation in Standard Relation

A compound can have multiple values in ChEMBL's raw data
1) A prioritizes the value types (IC50 > Ki > Kd > Inhibition)
2) Collect multiple values of the same type
3) Conditionally replaces them with the average value 
(IC50, Ki, Kd: if max - min <= 100 else drop)
(%Inhibition: if max - min <= 10 else drop)

For Custom data, a Custom_reader with only basic preprocessing is used
Basic process is also included in ChEMBL_reader

< Basic preprocess >
1) Remove missing values
2) Remove salts
3) Generate 'ROMOL' column from 'Smiles' column

The output is total, active and inactive data and is saved as .xlsx and .sdf
"""

# module info option
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ADis QSAR Data preprocessing')
    parser.add_argument('-d', '--data', required=True, help='input raw or preprocessed data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')
    parser.add_argument('-i', '--inhits', action='store_true', help='If you want add %Inhibition assays')
    parser.add_argument('-chembl', '--chembl', action='store_true', help='True if you use chembl data')
    args = parser.parse_args()

    # ignore warning
    pd.set_option('mode.chained_assignment',  None)

    # path
    path_data = Path(args.data)
    file_name = path_data.stem.split('_')[0]
    path_output = Path(args.output) / f'{file_name}_prepare'
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, 'prepare.log')

    # Start
    start = time.time()

    # initial info
    logger.info(f'Your data\t: {args.data}')
    logger.info(f'Output path\t: {args.output}')
    time.sleep(1)

    # load & preprocessing data
    filename_extension = path_data.suffix
    if args.chembl:

        # main_ar = int(input("Input active range (only int) [IC50, Ki, Kd] : "))
        # main_ir = int(input("Input inactive range (only int) [IC50, Ki, Kd] : "))
        main_ar = 100
        main_ir = 10000
        criteria = {'act': main_ar, 'inact': main_ir}
        if args.inhits:
            # criteria['i-inact'] = int(input("Input inactive range (only int) [%Inhibition] : "))
            criteria['i-inact'] = 20
        logger.info(f"Active range\t: {criteria}")

        logger.info(f'Read ChEMBL raw data..')
        c_reader = Reader.ChEMBL_reader()
        c_reader.run(filename_extension, criteria, path_data, args.inhits, path_output / file_name)
    else:
        logger.info(f'Read {path_data} data..')
        c_reader = Reader.Custom_reader()
        c_reader.run(filename_extension, path_data, path_output / file_name)

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")

