from pathlib import Path

path = Path('Vary_params')

for db in path.glob('*'):
    dn = db.stem
    if 'Summary' in dn:
        continue
    for fd in db.glob('*'):
        fn = fd.stem
        for ng in fd.glob('*'):
            nn = ng.stem
            for ep in ng.glob('*'):
                en = ep.suffix
                if en == '.tsv':
                    continue
                for sc in ep.glob('*'):
                    sn = sc.stem
                    for md in ['SVM', 'XGB', 'RF', 'MLP']:
                        md_file = sc / f"{fn}_model" / md / f"{fn}_{md}_model.pkl"
                        print(fn, nn, en, sn, md_file.stem)
                        md_file.unlink(missing_ok=True)
print('yes')