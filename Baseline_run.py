import sys
import subprocess
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from rdkit.Chem import PandasTools, AllChem
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

out_path = Path('Baseline_results')
out_path.mkdir(parents=True, exist_ok=True)


def fps(df, rd, bt):
    keys = [f"f_{x}" for x in range(bt)]
    PandasTools.AddMoleculeColumnToFrame(df)
    bit_generator = AllChem.GetMorganFingerprintAsBitVect
    df['fps'] = df['ROMol'].apply(lambda x: dict(zip(keys, np.array(bit_generator(x, useChirality=True, radius=rd, nBits=bt)))))

    new_df = []
    for _, row in df.iterrows():
        f_row = {'Compound_ID': row['Compound_ID'], 'AD': int(row['Active'])}
        f_row.update(row['fps'])
        new_df += [f_row]
    return pd.DataFrame(new_df)


data_path = Path(r"Dataset")
# scalers = {'Robust': RobustScaler(), 'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
scalers = {'Robust': RobustScaler()}
radius_type = {1: 'ECFP2', 2: 'ECFP4', 3: 'ECFP6'}

for db in data_path.glob('*'):
    dn = db.stem
    if 'Summary' in dn:
        continue
    for fd_idx, fd in enumerate(db.glob('*')):
        fn = fd.stem
        if 'table' in fn:
            continue
        g1 = pd.read_csv(fd / f"{fn}_g1.tsv", sep='\t')
        train = pd.read_csv(fd / f"{fn}_train.tsv", sep='\t')
        test = pd.read_csv(fd / f"{fn}_test.tsv", sep='\t')
        ext = pd.read_csv(fd / f"{fn}_external.tsv", sep='\t')

        # print(f'target : {fn}')
        # print('g1', len(g1))
        # print('train', len(train), train['Active'].value_counts()[1], train['Active'].value_counts()[0])
        # print('test', len(test), test['Active'].value_counts()[1], test['Active'].value_counts()[0])
        # print('ext', len(ext), ext['Active'].value_counts()[1], ext['Active'].value_counts()[0])
        # print('\n')

        for radius in [2]:
            for nbits in [256]:
                train_fps = fps(train, rd=radius, bt=nbits)
                test_fps = fps(test, rd=radius, bt=nbits)
                ext_fps = fps(ext, rd=radius, bt=nbits)

                fcols = [col for col in train_fps.columns if col.startswith('f_')]
                for s_type, scaler in scalers.items():
                    train_fps[fcols] = scaler.fit_transform(train_fps[fcols])
                    test_fps[fcols] = scaler.transform(test_fps[fcols])
                    ext_fps[fcols] = scaler.transform(ext_fps[fcols])

                    f_output = out_path / dn / fn / f"{radius_type[radius]}_{nbits}bits" / s_type
                    f_output.mkdir(parents=True, exist_ok=True)

                    scaler_path = (f_output / f"{fn}_scaler.pkl").as_posix()
                    train_path = (f_output / f"{fn}_train.tsv").as_posix()
                    test_path = (f_output / f"{fn}_test.tsv").as_posix()
                    ext_path = (f_output / f"{fn}_ext.tsv").as_posix()

                    # dump(scaler, scaler_path)
                    # train_fps.to_csv(train_path, sep='\t', index=False)
                    # test_fps.to_csv(test_path, sep='\t', index=False)
                    # ext_fps.to_csv(ext_path, sep='\t', index=False)

                    fwr = {'Dataset': fn, 'Type': f"{radius_type[radius]}_{nbits}bits"}

                    # generate model
                    for md in ['RF', 'XGB', 'SVM', 'MLP']:
                        # model_run = f'-train {train_path} -test {test_path} -ext {ext_path} -o {f_output.as_posix()} -m {md} -core 12'
                        # subprocess.run(args=[sys.executable, 'ADis_QSAR.py'] + model_run.split(' '))

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

                    with open(str(out_path / 'Baseline_Summary.tsv'), 'a') as fw:
                        if (fd_idx == 0) and (dn == 'ChEMBL'):
                            fw.write('\t'.join(fwr.keys()) + '\n')
                        fw.write('\t'.join(fwr.values()) + '\n')

