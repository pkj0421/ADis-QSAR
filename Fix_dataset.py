import sys
import subprocess
import pandas as pd
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def check_vals(df):
    act_dict = df['Active'].value_counts()
    return act_dict


out_path = Path('F_dataset')
out_path.mkdir(parents=True, exist_ok=True)

data_path = Path(r"Dataset")
for db in data_path.glob('*'):
    dn = db.stem
    if 'Summary' in dn:
        continue
    table = []

    if dn == 'ChEMBL':
        continue

    for fd_idx, fd in enumerate(db.glob('*')):
        fn = fd.stem
        if 'table' in fn:
            continue

        out_pth = out_path / dn / fn
        if dn == 'ChEMBL':
            raw_path = (fd / f"{fn}.tsv").as_posix()
            prepare_run = f'-d {raw_path} -o {out_pth.as_posix()} -i -chembl'
            subprocess.run(args=[sys.executable, 'Prepare.py'] + prepare_run.split(' '))

            act = out_pth / f"{fn}_prepare" / f"{fn}_active.tsv"
            inact = out_pth / f"{fn}_prepare" / f"{fn}_inactive.tsv"
            preprocess_run = f'-a {act.as_posix()} -i {inact.as_posix()} -o {out_pth.as_posix()} -core 12'
            subprocess.run(args=[sys.executable, 'Preprocessing.py'] + preprocess_run.split(' '))

        else:
            pre_g1 = pd.read_csv(fd / f"{fn}_g1.tsv", sep='\t')
            pre_train = pd.read_csv(fd / f"{fn}_train.tsv", sep='\t')
            pre_valid = pd.read_csv(fd / f"{fn}_test.tsv", sep='\t')
            pre_total = pd.concat([pre_g1, pre_train, pre_valid]).reset_index(drop=True)
            pre_total.to_csv(fd / f"{fn}_DUDE.tsv", sep='\t', index=False)

            raw_path = (fd / f"{fn}_DUDE.tsv").as_posix()
            prepare_run = f'-d {raw_path} -o {out_pth.as_posix()}'
            subprocess.run(args=[sys.executable, 'Prepare.py'] + prepare_run.split(' '))

            act = out_pth / f"{fn}_prepare" / f"{fn}_active.tsv"
            inact = out_pth / f"{fn}_prepare" / f"{fn}_inactive.tsv"
            test_pth = Path(rf'ChEMBL_Raw\{fn}.tsv').as_posix()
            preprocess_run = f'-a {act.as_posix()} -i {inact.as_posix()} -o {out_pth.as_posix()} -core 12 -t {test_pth}'
            subprocess.run(args=[sys.executable, 'Preprocessing.py'] + preprocess_run.split(' '))

        # check compounds
        g1 = pd.read_csv(out_path / dn / fn / f'{fn}_preprocessing' / f'{fn}_g1.tsv', sep='\t')
        train = pd.read_csv(out_path / dn / fn / f'{fn}_preprocessing' / f'{fn}_train.tsv', sep='\t')
        valid = pd.read_csv(out_path / dn / fn / f'{fn}_preprocessing' / f'{fn}_valid.tsv', sep='\t')
        test = pd.read_csv(out_path / dn / fn / f'{fn}_preprocessing' / f'{fn}_test.tsv', sep='\t')
        total = pd.concat([g1, train, valid, test]).reset_index(drop=True)
        for _, i in total.groupby('Smiles'):
            if len(i) >= 2:
                print(fn, 'duple!!!')

        ad = {'Total': check_vals(total),
              'G1': check_vals(g1),
              'Train': check_vals(train),
              'Valid': check_vals(valid),
              'Test': check_vals(test)}

        row = {'Source': dn, 'Target': fn,
               'Total': f"{len(total)} (Active {ad['Total'][1]} | Inactive {ad['Total'][0]})",
               'G1': f"{len(g1)} (Active {ad['G1'][1]})",
               'Train': f"{len(train)} (Active {ad['Train'][1]} | Inactive {ad['Train'][0]})",
               'Valid': f"{len(valid)} (Active {ad['Valid'][1]} | Inactive {ad['Valid'][0]})",
               'Test': f"{len(test)} (Active {ad['Test'][1]} | Inactive {ad['Test'][0]})"}
        table += [row]

    table_df = pd.DataFrame(table)
    table_df.to_csv(out_path / dn / f'{dn}_table.tsv', sep='\t', index=False)


