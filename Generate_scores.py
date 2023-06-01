
import pandas as pd
import multiprocessing

from pathlib import Path
from rdkit.Chem import PandasTools, AllChem
from Admodule.Grouping import Vector


def fps(df, rd, bt, g1, n_cpu):
    PandasTools.AddMoleculeColumnToFrame(df)
    PandasTools.AddMoleculeColumnToFrame(g1)
    bit_generator = AllChem.GetMorganFingerprintAsBitVect
    df['bits'] = df['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))
    g1['bits'] = g1['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt))

    # vectorize
    vectorize = Vector()
    vector = vectorize.run(g1, df, n_cpu, ss=False)
    return vector


if __name__ == '__main__':
    path = Path("Dataset")
    out_path = Path("Score_distribution")
    out_path.mkdir(parents=True, exist_ok=True)

    for db in path.glob('*'):
        dn = db.stem
        if 'TST' in dn:
            continue

        for fd in db.glob('*'):
            fn = fd.stem
            if 'dataset' in fn:
                continue

            g1 = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_g1.tsv", sep='\t')
            train = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_train.tsv", sep='\t')
            valid = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_valid.tsv", sep='\t')
            test = pd.read_csv(fd / f"{fn}_preprocessing" / f"{fn}_test.tsv", sep='\t')

            radius = 2
            nbits = 256
            cores = multiprocessing.cpu_count()
            train_vector = fps(train, rd=radius, bt=nbits, g1=g1, n_cpu=cores)
            valid_vector = fps(valid, rd=radius, bt=nbits, g1=g1, n_cpu=cores)
            test_vector = fps(test, rd=radius, bt=nbits, g1=g1, n_cpu=cores)

            f_output = out_path / dn / fn
            f_output.mkdir(parents=True, exist_ok=True)

            train_vector.to_csv(f_output / f"{fn}_train_scores.tsv", sep='\t', index=False)
            valid_vector.to_csv(f_output / f"{fn}_valid_scores.tsv", sep='\t', index=False)
            test_vector.to_csv(f_output / f"{fn}_test_scores.tsv", sep='\t', index=False)

            # fcols = [col for col in train_vector.columns if col.startswith('f_')]



