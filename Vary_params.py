import sys
import time
import itertools
import subprocess
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from rdkit.Chem import PandasTools, AllChem
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from Admodule.Grouping import Cluster, Vector


from knockknock import discord_sender
# notice discord
webhook_url = "https://discord.com/api/webhooks/1009749385170665533/m4nldXOXR5f9iWaXoCDLNGhNI48XEpy-Y9CcBpdFJW_xUipS54LCzXX9xZaCY6IH0vSl"
@discord_sender(webhook_url=webhook_url)
def finish(message):
    return message  # Optional return value

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

out_path = Path('Vary_params_results')
out_path.mkdir(parents=True, exist_ok=True)


def fps(df, rd, bt, g1):
    PandasTools.AddMoleculeColumnToFrame(df)
    bit_generator = AllChem.GetMorganFingerprintAsBitVect
    df['bits'] = df['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))
    g1['bits'] = g1['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))

    # vectorize
    vector = vectorize.run(g1, df, 12)
    return vector


if __name__ == "__main__":
    data_path = Path(r"Dataset")
    scalers = {'Robust': RobustScaler(), 'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    radius_type = {1: 'ECFP2', 2: 'ECFP4', 3: 'ECFP6'}

    clt = Cluster()
    vectorize = Vector()
    run_list = ['IRAK4', 'SYK', 'CSF1R', 'KPCB', 'AKT1', 'FAK1', 'MET']

    for db in data_path.glob('*'):
        dn = db.stem
        if 'Summary' in dn:
            continue
        for fd_idx, fd in enumerate(db.glob('*')):
            fn = fd.stem
            if 'table' in fn:
                continue
            if fn not in run_list:
                continue

            start = time.time()
            for g1_cnt in [10, 50, 100]:
                fdata_path = out_path / dn / fn / f"nG1_{g1_cnt}"
                fdata_path.mkdir(parents=True, exist_ok=True)
                total = pd.read_csv(fd / f"{fn}_total.tsv", sep='\t')
                PandasTools.AddMoleculeColumnToFrame(total)

                test = total[total['Set_type'] == 'Test']
                ext = total[total['Set_type'] == 'External']

                if g1_cnt == 50:
                    train = total[total['Set_type'] == 'Train']
                    g1 = total[total['Set_type'] == 'G1']
                    g1.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_g1.tsv", sep='\t', index=False)
                    train.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_train.tsv", sep='\t', index=False)
                    test.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_test.tsv", sep='\t', index=False)
                    ext.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_external.tsv", sep='\t', index=False)

                else:
                    train = total[(total['Set_type'] == 'Train') | (total['Set_type'] == 'G1')]
                    train_act = train[train['Active'] == 1]
                    train_inact = train[train['Active'] == 0]
                    train_inact['Set'] = 'G2 inactive'

                    g1_cls = True
                    rb_idx = 0
                    rb_lst = [x for x in itertools.product([3, 2, 1], [2048, 1024, 512, 256])]
                    while g1_cls:
                        if rb_idx == len(rb_lst) - 1:
                            break
                        g1, g1_remains = clt.run(train_act, g1_cnt, rb_lst[rb_idx], 12)
                        if len(g1) == g1_cnt:
                            g1_cls = False
                        rb_idx += 1

                    if len(g1) > g1_cnt:
                        g1.reset_index(drop=True, inplace=True)
                        num_rows = len(g1) - g1_cnt
                        random_rows = g1.sample(num_rows)

                        g1.drop(random_rows.index, inplace=True)
                        g1_remains = pd.concat([g1_remains, random_rows])
                        g1_remains.reset_index(drop=True, inplace=True)

                    g1_remains['Set'] = 'G2 active'
                    train = pd.concat([g1_remains, train_inact])

                    g1['Set'] = 'G1 active'
                    g1['Set_type'] = 'G1'
                    train['Set_type'] = 'Train'

                    g1.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_g1.tsv", sep='\t', index=False)
                    train.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_train.tsv", sep='\t', index=False)
                    test.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_test.tsv", sep='\t', index=False)
                    ext.drop(columns='ROMol').to_csv(fdata_path / f"{fn}_external.tsv", sep='\t', index=False)

                for radius in [1, 2, 3]:
                    for nbits in [256, 512]:

                        if g1_cnt == 50 and radius == 2 and nbits == 256:
                            continue

                        train_vector = fps(train, rd=radius, bt=nbits, g1=g1)
                        test_vector = fps(test, rd=radius, bt=nbits, g1=g1)
                        ext_vector = fps(ext, rd=radius, bt=nbits, g1=g1)

                        fcols = [col for col in train_vector.columns if col.startswith('f_')]
                        for s_type, scaler in scalers.items():

                            # scaling
                            train_vector[fcols] = scaler.fit_transform(train_vector[fcols])
                            test_vector[fcols] = scaler.transform(test_vector[fcols])
                            ext_vector[fcols] = scaler.transform(ext_vector[fcols])

                            # save
                            f_output = fdata_path / f"{radius_type[radius]}_{nbits}bits" / s_type
                            f_output.mkdir(parents=True, exist_ok=True)

                            scaler_path = (f_output / f"{fn}_scaler.pkl").as_posix()
                            train_path = (f_output / f"{fn}_train_vector.tsv").as_posix()
                            test_path = (f_output / f"{fn}_test_vector.tsv").as_posix()
                            ext_path = (f_output / f"{fn}_ext_vector.tsv").as_posix()

                            dump(scaler, scaler_path)
                            train_vector.to_csv(train_path, sep='\t', index=False)
                            test_vector.to_csv(test_path, sep='\t', index=False)
                            ext_vector.to_csv(ext_path, sep='\t', index=False)

                            fwr = {'Target': fn, 'nG1': g1_cnt, 'Fingerprint_type': f"{radius_type[radius]}_{nbits}bits", 'Scaler': s_type}

                            # generate model
                            for md in ['RF', 'XGB', 'SVM', 'MLP']:
                                model_run = f'-train {train_path} -test {test_path} -ext {ext_path} -o {f_output.as_posix()} -m {md} -core 12'
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

                            with open(str(out_path / 'Vary_params_Summary.tsv'), 'a') as fw:
                                if (fd_idx == 0) and (dn == 'ChEMBL'):
                                    fw.write('\t'.join(fwr.keys()) + '\n')
                                fw.write('\t'.join(fwr.values()) + '\n')

            f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            finish(f'\nFinished: {fn}\nLearning time: {f_time}')
