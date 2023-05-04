import time
import logging
import warnings
import argparse
import pandas as pd

from joblib import dump
from pathlib import Path
from Admodule import Reader, Grouping, Utils
from sklearn.preprocessing import RobustScaler

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing data')
    parser.add_argument('-a', '--active', required=True, help='Active data')
    parser.add_argument('-i', '--inactive', required=True, help='Inactive data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')
    parser.add_argument('-t', '--test_size', type=float, default=0.2, help='Set your test size')
    parser.add_argument('-r', '--radius', type=int, default=2, help='Set your radius')
    parser.add_argument('-b', '--bits', type=int, default=256, help='Set your nbits')
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
    logger.info(f'Test size : {args.test_size}')
    logger.info(f'Fingerprint radius : {args.radius}')
    logger.info(f'Fingerprint nbits : {args.bits}')
    logger.info(f'Use cores : {n_cores}')

    # load data
    c_reader = Reader.Custom_reader()
    active = c_reader.run(path_active.suffix, path_active)
    inactive = c_reader.run(path_inactive.suffix, path_inactive)

    # # split train & test
    train_size = 10 - (args.test_size * 10)
    test_size = args.test_size * 10
    clt = Grouping.Cluster()

    rb = [args.radius, args.bits]
    g1, g1_remains = clt.run(active, 50, rb, n_cores)

    # Grouping
    g1['Group'] = 'G1'
    g1_remains['Group'] = 'G2'
    inactive['Group'] = 'G2'

    # divide active
    da = len(g1_remains) / 10
    tra, tea = list(map(round, [da * train_size, da * test_size]))
    train_act, test_act = clt.run(g1_remains, tra, rb, n_cores)

    # divide inactive
    di = len(inactive) / 10
    tri, tei = list(map(round, [da * train_size, da * test_size]))
    train_inact, test_inact = clt.run(inactive, tri, rb, n_cores)

    # save dataset
    g2_train = pd.concat([train_act, train_inact])
    g2_test = pd.concat([test_act, test_inact])
    Utils.save(g1, path_output / file_name, custom=f"g1")
    Utils.save(g2_train, path_output / file_name, custom=f"train")
    Utils.save(g2_test, path_output / file_name, custom=f"test")

    # vectorize
    vectorize = Grouping.Vector()
    logger.info('Generate train vectors...')
    train_vector = vectorize.run(g1, g2_train, n_cores)
    logger.info('Generate test vectors...')
    test_vector = vectorize.run(g1, g2_test, n_cores)

    # scaling
    logger.info('Scaling vectors...')
    scaler = RobustScaler()
    cols = [col for col in train_vector.columns if col.startswith('f_')]
    train_vector[cols] = scaler.fit_transform(train_vector[cols])
    test_vector[cols] = scaler.transform(test_vector[cols])

    # save
    dump(scaler, path_output / f"{file_name}_scaler.pkl")
    Utils.save(train_vector, path_output / file_name, custom=f"train_vector")
    Utils.save(test_vector, path_output / file_name, custom=f"test_vector")

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")

