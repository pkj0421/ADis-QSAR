import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem import PandasTools, AllChem

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
out_path = Path('Baseline_noscale')
out_path.mkdir(parents=True, exist_ok=True)
radius_type = {1: 'ECFP2', 2: 'ECFP4', 3: 'ECFP6'}

col_wr = True
for db in data_path.glob('*'):
    dn = db.stem
    if 'Summary' in dn:
        continue

    for fd_idx, fd in enumerate(db.glob('*')):
        fn = fd.stem
        if 'table' in fn:
            continue

        g1 = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_g1.tsv", sep='\t')
        train = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_train.tsv", sep='\t')
        valid = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_valid.tsv", sep='\t')
        test = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_test.tsv", sep='\t')

        for radius in [2, 3]:
            for nbits in [256, 512]:
                train_fps = fps(train, rd=radius, bt=nbits)
                valid_fps = fps(valid, rd=radius, bt=nbits)
                test_fps = fps(test, rd=radius, bt=nbits)

                t_output = out_path / dn / fn / f"{radius_type[radius]}_{nbits}bits"
                t_output.mkdir(parents=True, exist_ok=True)

                train_raw_path = (t_output / f"{fn}_raw_train.tsv").as_posix()
                valid_raw_path = (t_output / f"{fn}_raw_valid.tsv").as_posix()
                test_raw_path = (t_output / f"{fn}_raw_test.tsv").as_posix()

                train_fps.to_csv(train_raw_path, sep='\t', index=False)
                valid_fps.to_csv(valid_raw_path, sep='\t', index=False)
                test_fps.to_csv(test_raw_path, sep='\t', index=False)

                fwr = {'Dataset': fn, 'Type': f"{radius_type[radius]}_{nbits}bits"}

                # generate model
                for md in ['RF', 'XGB', 'SVM', 'MLP']:
                    model_run = f'-train {train_raw_path} -valid {valid_raw_path} -test {test_raw_path} -o {t_output.as_posix()} -m {md} -core 12'
                    subprocess.run(args=[sys.executable, 'ADis_QSAR.py'] + model_run.split(' '))

                    model_path = t_output / f"{fn}_model" / md
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

                with open(str(out_path / f'{out_path.stem}_Summary.tsv'), 'a') as fw:
                    if col_wr:
                        fw.write('\t'.join(fwr.keys()) + '\n')
                        col_wr = False
                    fw.write('\t'.join(fwr.values()) + '\n')

