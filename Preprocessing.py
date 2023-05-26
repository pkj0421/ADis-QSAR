import time
import logging
import warnings
import argparse
import itertools
import pandas as pd

from joblib import dump
from pathlib import Path
from rdkit.Chem import AllChem, CanonSmiles
from Admodule import Reader, Grouping, Utils
from Admodule.Reader import ChEMBL_reader
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

"""
This code is used for data preprocessing
Need active and inactive compounds data
ADis-QSAR aims for an active:inactive ratio of 1:~1.5, excluding 50 central structures (actives)

Pair system is applied to generate the vectors required for model training
And clustering maximizes the chemical space of the data

< Preprocessing >
1) Clustering
Group_1 : 50 central structures that can represent all active structures
Group_2 : Others (remain active structures + inactive structures)
2) Pairing
Group_1 X Group_2 (Compare fingerprint bits & Pre-scoring)
4) Collect paired data group based on Group_2
5) Sum all pre-scores for each bit position in the group

The output is train(sdf, vector), test(sdf, vector), and scaler(.pkl) files
"""

# module info option
logger = logging.getLogger(__name__)


def adjust_ratio(df1, df2, ratio, cores):
    len1 = df1.shape[0]
    len2 = df2.shape[0]

    if len1 > len2:
        target_len = int(len2 * (1 / ratio))
        df1_cent = pick_molecules(df1, target_len, cores)[0]
        df2_cent = df2.copy()

    elif len2 >= len1:
        target_len = int(len1 * ratio)
        if len2 == target_len:
            df1_cent = df1.copy()
            df2_cent = df2.copy()
        elif len2 > target_len:
            df1_cent = df1.copy()
            df2_cent = pick_molecules(df2, target_len, cores)[0]
        else:
            fix_target_len = int(len2 * (1 / ratio))
            df1_cent = pick_molecules(df1, fix_target_len, cores)[0]
            df2_cent = df2.copy()

    return df1_cent, df2_cent


def pick_molecules(df, cls_num, cores, rb=False):
    rb_idx = 0
    check = True
    rb_lst = [x for x in itertools.product([3, 2, 1], [2048, 1024, 512, 256])]
    clt = Grouping.Cluster()

    while check:
        if rb_idx == len(rb_lst) - 1:
            break
        logger.info(f'Use radius, nbits for Butina clustering : {rb_lst[rb_idx]}')
        cent, remains = clt.run(df, cls_num, rb_lst[rb_idx], cores)
        if len(cent) == cls_num:
            check = False
        rb_idx += 1

    if len(cent) > cls_num:
        cent.reset_index(drop=True, inplace=True)
        num_rows = len(cent) - cls_num
        random_rows = cent.sample(num_rows, random_state=42)

        cent.drop(random_rows.index, inplace=True)
        remains = pd.concat([remains, random_rows])
        remains.reset_index(drop=True, inplace=True)
    elif len(cent) < cls_num:
        cent.reset_index(drop=True, inplace=True)
        num_rows = cls_num - len(cent)
        random_rows = remains.sample(num_rows, random_state=42)

        remains.drop(random_rows.index, inplace=True)
        cent = pd.concat([cent, random_rows])
        cent.reset_index(drop=True, inplace=True)
    else:
        pass

    if rb:
        bit_generator = AllChem.GetMorganFingerprintAsBitVect
        cent['bits'] = cent['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rb[0], nBits=rb[1]))
        remains['bits'] = remains['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rb[0], nBits=rb[1]))
    return cent, remains


def duple_structures(df1, df2):
    df1['canonical_smi'] = df1['Smiles'].apply(CanonSmiles)
    df2['canonical_smi'] = df2['Smiles'].apply(CanonSmiles)
    remove_duple = df1[~df1['canonical_smi'].isin(df2['canonical_smi'])]
    return remove_duple



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing data')
    parser.add_argument('-a', '--active', required=True, help='Active data')
    parser.add_argument('-i', '--inactive', required=True, help='Inactive data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')
    parser.add_argument('-v', '--valid_size', type=float, default=0.2, help='Set your valid size')
    parser.add_argument('-t', '--test_set', type=str, default='X', help='Set your test size')
    parser.add_argument('-r', '--radius', type=int, default=2, help='Set your radius')
    parser.add_argument('-b', '--bits', type=int, default=256, help='Set your nbits')
    parser.add_argument('-s', '--scaler', type=str, default='Robust', help='Set your scaler')
    parser.add_argument('-core', '--num_cores', type=int, default=2, help='Set the number of CPU cores to use')
    args = parser.parse_args()

    # ignore warning
    pd.set_option('mode.chained_assignment', None)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # path
    path_active = Path(args.active)
    path_inactive = Path(args.inactive)
    file_name = path_active.stem.split('_')[0]
    path_output = Path(args.output) / f"{file_name}_preprocessing"
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, 'preprocess.log')

    # Start
    start = time.time()

    # set cores
    n_cores = Utils.set_cores(args.num_cores)

    # initial info
    logger.info(f'Active data : {path_active}')
    logger.info(f'Inactive data : {path_inactive}')
    logger.info(f'Output path : {path_output}')
    logger.info(f'Fingerprint radius : {args.radius}')
    logger.info(f'Fingerprint nbits : {args.bits}')
    logger.info(f'Use cores : {n_cores}')

    # load data
    c_reader = Reader.Custom_reader()
    active = c_reader.run(path_active.suffix, path_active)
    inactive = c_reader.run(path_inactive.suffix, path_inactive)

    # # split train & valid & test
    train_size = 10 - (args.valid_size * 10)
    valid_size = args.valid_size * 10
    test_size = valid_size
    logger.info(f'Train : Valid = {int(train_size)} : {int(valid_size)}')
    logger.info(f"Test size = {int(test_size)}")

    rb = [args.radius, args.bits]
    g1, g1_remains = pick_molecules(active, 50, n_cores, rb)

    # Grouping
    g1['Group'] = 'G1'
    g1_remains['Group'] = 'G2'
    inactive['Group'] = 'G2'

    # set active : inactive ratio
    g1_remains, inactive = adjust_ratio(g1_remains, inactive, 1.5, n_cores)
    logger.info(f"Adjust ratio : active ({len(g1_remains)}), inactive ({len(inactive)})")

    if args.test_set == 'X':
        # divide active
        total_ratio = train_size + valid_size + test_size
        da = len(g1_remains) / total_ratio
        tra, va, tea = list(map(round, [da * train_size, da * valid_size, da * test_size]))
        logger.info(f"tra : {tra}, va : {va}, tea : {tea}")
        train_act, ta_remains = pick_molecules(g1_remains, tra, n_cores, rb)
        valid_act, test_act = pick_molecules(ta_remains, va, n_cores, rb)
        logger.info(f"train_act : {len(train_act)}, valid_act : {len(valid_act)}, test_act : {len(test_act)}")

        # divide inactive
        di = len(inactive) / total_ratio
        tri, vi, tei = list(map(round, [di * train_size, di * valid_size, di * test_size]))
        logger.info(f"tri : {tri}, vi : {vi}, tei : {tei}")

        train_inact, ti_remains = pick_molecules(inactive, tri, n_cores, rb)
        valid_inact, test_inact = pick_molecules(ti_remains, vi, n_cores, rb)
        logger.info(f"train_inact : {len(train_inact)}, valid_inact : {len(valid_inact)}, test_inact : {len(test_inact)}")

    else:
        c_reader = ChEMBL_reader()
        criteria = {'act': 1000, 'inact': 30000, 'i-inact': 20}
        test_set = c_reader.run(".tsv", criteria, Path(rf"{args.test_set}"))

        pre_total = pd.concat([active, inactive]).reset_index(drop=True)
        pre_test = duple_structures(test_set, pre_total)
        logger.info(f"Remove duple structure in test set : {len(test_set)} -> {len(pre_test)}")

        pre_test_act = pre_test[pre_test['Active'] == 1].reset_index(drop=True)
        pre_test_inact = pre_test[pre_test['Active'] == 0].reset_index(drop=True)

        # divide active
        total_ratio = train_size + valid_size
        da = len(g1_remains) / total_ratio
        tra, va = list(map(round, [da * train_size, da * valid_size]))
        logger.info(f"tra : {tra}, va : {va}, tea : {va}")

        train_act, ta_remains = pick_molecules(g1_remains, tra, n_cores, rb)
        valid_act = pick_molecules(ta_remains, va, n_cores, rb)[0]
        test_act = pick_molecules(pre_test_act, va, n_cores, rb)[0]
        logger.info(f"train_act : {len(train_act)}, valid_act : {len(valid_act)}, test_act : {len(test_act)}")

        # divide inactive
        di = len(inactive) / total_ratio
        tri, vi = list(map(round, [di * train_size, di * valid_size]))
        logger.info(f"tri : {tri}, vi : {vi}, tei : {vi}")

        train_inact, ti_remains = pick_molecules(inactive, tri, n_cores, rb)
        valid_inact = pick_molecules(ti_remains, vi, n_cores, rb)[0]
        test_inact = pick_molecules(pre_test_inact, vi, n_cores, rb)[0]
        logger.info(f"train_inact : {len(train_inact)}, valid_inact : {len(valid_inact)}, test_inact : {len(test_inact)}")

    # save dataset
    g2_train = pd.concat([train_act, train_inact]).reset_index(drop=True)
    g2_valid = pd.concat([valid_act, valid_inact]).reset_index(drop=True)
    g2_test = pd.concat([test_act, test_inact]).reset_index(drop=True)
    logger.info(f"G1 : {len(g1)}, Train : {len(g2_train)}, Valid : {len(g2_valid)}, Test : {len(g2_test)}")

    Utils.save(g1, path_output / file_name, custom=f"g1")
    Utils.save(g2_train, path_output / file_name, custom=f"train")
    Utils.save(g2_valid, path_output / file_name, custom=f"valid")
    Utils.save(g2_test, path_output / file_name, custom=f"test")

    # vectorize
    vectorize = Grouping.Vector()
    logger.info('Generate train vectors...')
    train_vector = vectorize.run(g1, g2_train, n_cores)
    logger.info('Generate valid vectors...')
    valid_vector = vectorize.run(g1, g2_valid, n_cores)
    logger.info('Generate test vectors...')
    test_vector = vectorize.run(g1, g2_test, n_cores)

    # save before scaling
    Utils.save(train_vector, path_output / file_name, custom=f"train_raw_vector")
    Utils.save(valid_vector, path_output / file_name, custom=f"valid_raw_vector")
    Utils.save(test_vector, path_output / file_name, custom=f"test_raw_vector")

    # scaling
    logger.info('Scaling vectors...')
    scalers = {'Robust': RobustScaler(), 'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    scaler = scalers[args.scaler]
    cols = [col for col in train_vector.columns if col.startswith('f_')]
    train_vector[cols] = scaler.fit_transform(train_vector[cols])
    valid_vector[cols] = scaler.transform(valid_vector[cols])
    test_vector[cols] = scaler.transform(test_vector[cols])

    # save
    dump(scaler, path_output / f"{file_name}_{args.scaler}_scaler.pkl")
    Utils.save(train_vector, path_output / file_name, custom=f"train_vector")
    Utils.save(valid_vector, path_output / file_name, custom=f"valid_vector")
    Utils.save(test_vector, path_output / file_name, custom=f"test_vector")

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")

