import sys
import time
import itertools
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from joblib import dump
from pathlib import Path
from rdkit.Chem import PandasTools, AllChem
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from Admodule.Grouping import Cluster, Vector

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def fps(df, rd, bt, g1, n_cpu):
    PandasTools.AddMoleculeColumnToFrame(df)
    bit_generator = AllChem.GetMorganFingerprintAsBitVect
    df['bits'] = df['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))
    g1['bits'] = g1['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))

    # vectorize
    vector = vectorize.run(g1, df, n_cpu)
    return vector


def pick_molecules(df, cls_num, cores):
    rb_idx = 0
    check = True
    rb_lst = [x for x in itertools.product([3, 2, 1], [2048, 1024, 512, 256])]
    clt = Cluster()

    while check:
        if rb_idx == len(rb_lst) - 1:
            break
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

    return cent, remains


if __name__ == "__main__":
    data_path = Path(r"Dataset")
    out_path = Path('Vary_params_results')
    out_path.mkdir(parents=True, exist_ok=True)
    scalers = {'Robust': RobustScaler(), 'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    radius_type = {1: 'ECFP2', 2: 'ECFP4', 3: 'ECFP6'}

    clt = Cluster()
    vectorize = Vector()
    cores = multiprocessing.cpu_count() - 1

    col_wr = True
    for db in data_path.glob('*'):
        dn = db.stem
        if 'TST' in dn:
            continue

        for fd_idx, fd in enumerate(db.glob('*')):
            fn = fd.stem
            if 'dataset' in fn:
                continue

            start = time.time()
            for g1_cnt in [20, 50, 80]:

                fdata_path = out_path / dn / fn / f"nG1_{g1_cnt}"
                fdata_path.mkdir(parents=True, exist_ok=True)

                g1 = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_g1.tsv", sep='\t')
                train = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_train.tsv", sep='\t')
                valid = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_valid.tsv", sep='\t')
                test = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_test.tsv", sep='\t')

                PandasTools.AddMoleculeColumnToFrame(g1)
                PandasTools.AddMoleculeColumnToFrame(train)
                PandasTools.AddMoleculeColumnToFrame(valid)
                PandasTools.AddMoleculeColumnToFrame(test)

                if g1_cnt == 50:

                    g1.to_csv(fdata_path / f"{fn}_g1.tsv", sep='\t', index=False)
                    train.to_csv(fdata_path / f"{fn}_train.tsv", sep='\t', index=False)
                    valid.to_csv(fdata_path / f"{fn}_valid.tsv", sep='\t', index=False)
                    test.to_csv(fdata_path / f"{fn}_test.tsv", sep='\t', index=False)

                else:
                    train_total = pd.concat([g1, train]).reset_index(drop=True)
                    train_act = train_total[train_total['Active'] == 1]
                    train_inact = train_total[train_total['Active'] == 0]
                    train_inact['Set'] = 'G2 inactive'

                    g1, g1_remains = pick_molecules(train_act, g1_cnt, cores)
                    g1_remains['Set'] = 'G2 active'
                    train = pd.concat([g1_remains, train_inact])

                    g1['Set'] = 'G1 active'
                    g1['Set_type'] = 'G1'
                    train['Set_type'] = 'Train'

                    g1.to_csv(fdata_path / f"{fn}_g1.tsv", sep='\t', index=False)
                    train.to_csv(fdata_path / f"{fn}_train.tsv", sep='\t', index=False)
                    valid.to_csv(fdata_path / f"{fn}_valid.tsv", sep='\t', index=False)
                    test.to_csv(fdata_path / f"{fn}_test.tsv", sep='\t', index=False)

                for radius in [2, 3]:
                    for nbits in [256, 512]:

                        train_vector = fps(train, rd=radius, bt=nbits, g1=g1, n_cpu=cores)
                        valid_vector = fps(valid, rd=radius, bt=nbits, g1=g1, n_cpu=cores)
                        test_vector = fps(test, rd=radius, bt=nbits, g1=g1, n_cpu=cores)

                        fcols = [col for col in train_vector.columns if col.startswith('f_')]
                        for s_type, scaler in scalers.items():

                            # scaling
                            train_vector[fcols] = scaler.fit_transform(train_vector[fcols])
                            valid_vector[fcols] = scaler.transform(valid_vector[fcols])
                            test_vector[fcols] = scaler.transform(test_vector[fcols])

                            # save
                            f_output = fdata_path / f"{radius_type[radius]}_{nbits}bits" / s_type
                            f_output.mkdir(parents=True, exist_ok=True)

                            scaler_path = (f_output / f"{fn}_{s_type}_scaler.pkl").as_posix()
                            train_path = (f_output / f"{fn}_train_vector.tsv").as_posix()
                            valid_path = (f_output / f"{fn}_valid_vector.tsv").as_posix()
                            test_path = (f_output / f"{fn}_test_vector.tsv").as_posix()

                            dump(scaler, scaler_path)
                            train_vector.to_csv(train_path, sep='\t', index=False)
                            valid_vector.to_csv(valid_path, sep='\t', index=False)
                            test_vector.to_csv(test_path, sep='\t', index=False)

                            fwr = {'Target': fn, 'nG1': g1_cnt, 'Fingerprint_type': f"{radius_type[radius]}_{nbits}bits", 'Scaler': s_type}

                            # generate model
                            for md in ['RF', 'XGB', 'SVM', 'MLP']:
                                model_run = f'-train {train_path} -valid {valid_path} -test {test_path} -o {f_output.as_posix()} -m {md} -core {cores}'
                                subprocess.run(args=[sys.executable, 'ADis_QSAR.py'] + model_run.split(' '))

                                model_path = f_output / f"{fn}_model" / md
                                mcs = pd.read_csv(model_path / f"{fn}_{md}_model_score_log.tsv", sep='\t')

                                ACC = []
                                AUC = []
                                PR = []
                                SP = []
                                for mcd in mcs.to_dict('records'):
                                    mcd_name = mcd['Data'].capitalize()
                                    acc = f"{mcd_name} {float(mcd['ACC'].split(' ')[0]):.2f}"
                                    auc = f"{mcd_name} {float(mcd['AUC']):.2f}"
                                    pr = f"{mcd_name} {float(mcd['Precision'].split(' ')[0]):.2f}"
                                    sp = f"{mcd_name} {float(mcd['Specificity'].split(' ')[0]):.2f}"

                                    ACC += [acc]
                                    AUC += [auc]
                                    PR += [pr]
                                    SP += [sp]

                                ACC = ' | '.join(ACC)
                                AUC = ' | '.join(AUC)
                                PR = ' | '.join(PR)
                                SP = ' | '.join(SP)

                                fwr[f"{md} ACC"] = ACC
                                fwr[f"{md} AUC"] = AUC
                                fwr[f"{md} PR"] = PR
                                fwr[f"{md} SP"] = SP

                            with open(str(out_path / f'Summary_of_{out_path.stem}_results.tsv'), 'a') as fw:
                                if col_wr:
                                    fw.write('\t'.join(fwr.keys()) + '\n')
                                    col_wr = False
                                fw.write('\t'.join(map(str, fwr.values())) + '\n')

            f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            print(f'\nFinished: {fn}\nLearning time: {f_time}')
