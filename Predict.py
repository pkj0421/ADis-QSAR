import time
import logging
import argparse
import numpy as np
import pandas as pd

from joblib import load
from pathlib import Path
from rdkit.Chem import AllChem
from Admodule import Reader, Grouping, Utils
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score

"""
This code is used for predict external data
Need g1 dataset(.xlsx), external dataset(.xlsx), scaler(.pkl), model(.pkl) files
If the vector of external data has already been generated, it can be directly used in the ADis-QSAR code.
The output is external vector(.tsv), predict log file(.tsv)
"""

# module info option
logger = logging.getLogger(__name__)


def predict_ext(md, ext):
    x_cols = [col for col in ext.columns if col.startswith('f_')]
    y_pred = md.predict(ext[x_cols])
    result = pd.DataFrame({'Compound_ID': ext['Compound_ID'], 'Pred': y_pred})
    sort_result = result.sort_values('Pred', ascending=False)
    logger.info(f"Predict active structure from external set : {sort_result['Pred'].sum()}")
    return sort_result


def confusion_matrix_scorer(clf, ext, output):
    x_cols = [col for col in ext.columns if col.startswith('f_')]
    X = ext[x_cols]
    y = ext['AD']

    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    auc = round(roc_auc_score(y, y_pred), 2)
    r2 = round(r2_score(y, y_pred), 2)

    result = pd.DataFrame({'Compound_ID': ext['Compound_ID'], 'Active': y, 'Pred': y_pred})
    result['Match'] = result[['Active', 'Pred']].apply(lambda x: 1 if x[0] == x[1] else 0, axis=1)
    result.to_excel(output, sheet_name='Sheet1', index=False, header=True)

    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'auc': auc, 'r2': r2}


def result_scoring(md, ext, name, out):
    output = out / f"{name}_predictions.xlsx"
    cm = confusion_matrix_scorer(md, ext, output)
    if cm['tp'] != 0:
        precision = round(cm['tp'] / (cm['tp'] + cm['fp']), 2)
        recall = round(cm['tp'] / (cm['tp'] + cm['fn']), 2)
        f1 = round(2 * (recall * precision) / (recall + precision), 2)
    else:
        precision = 0
        recall = 0
        f1 = 0

    if cm['tn'] != 0:
        specificity = round(cm['tn'] / (cm['tn'] + cm['fp']), 2)
    else:
        specificity = 0

    answer = cm['tp'] + cm['tn']
    if answer != 0:
        accuracy = round(answer / len(ext), 2)
    else:
        accuracy = 0

    result = {'Data': name,
              'ACC': f"{accuracy} ({answer} / {len(ext)})",
              'TP': cm['tp'], 'FP': cm['fp'], 'FN': cm['fn'], 'TN': cm['tn'],
              'Precision': f"{precision} ({cm['tp']} / {cm['tp']} + {cm['fp']})",
              'Recall': f"{recall} ({cm['tp']} / {cm['tp']} + {cm['fn']})",
              'F1': f"{f1} (2 * ({recall} * {precision}) / {recall} + {precision})",
              'Specificity': f"{specificity} ({cm['tn']} / {cm['tn']} + {cm['fp']})",
              'AUC': cm['auc'], 'r2': cm['r2']}

    logger.info(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict external data')
    parser.add_argument('-m', '--model', required=True, help='Set your model')
    parser.add_argument('-e', '--external', required=True, help='Set external data to predict')
    parser.add_argument('-n', '--name', type=str, required=True, help='Set your external data name')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')
    parser.add_argument('-core', '--num_cores', type=int, default=2, help='Set the number of CPU cores to use')
    parser.add_argument('-ev', '--ev', action='store_true', help='If you already have ext vectors..')

    # if ps == True:
    parser.add_argument('-g1', '--g1', default=False, help='Set g1 set was used to generate model')
    parser.add_argument('-s', '--scaler', default=False, help='Set data scaler')
    parser.add_argument('-r', '--radius', type=int, default=2, help='Set your radius')
    parser.add_argument('-b', '--bits', type=int, default=1024, help='Set your nbits')

    args = parser.parse_args()

    # Start
    start = time.time()

    # set cores
    n_cores = Utils.set_cores(args.num_cores)

    # path
    path_ext = Path(args.external)
    path_model = Path(args.model)
    file_name = path_model.stem.split('_')[0]
    path_output = Path(args.output) / f'{file_name}_predict'
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, 'predict.log')

    # load data
    model = load(str(path_model))

    # initial info
    logger.info('Predict external set...')
    logger.info(f"Load model :\n{model}")

    if not args.ev:
        # read compounds
        c_reader = Reader.Custom_reader()
        path_g1 = Path(args.g1)
        path_scaler = Path(args.scaler)
        g1 = c_reader.run(path_g1.suffix, path_g1)
        ext = c_reader.run(path_ext.suffix, path_ext)
        logger.info(f"G1 : {len(g1)} | External : {len(ext)}")

        # generate fingerprints (use the same parameters as the train set)
        bit_generator = AllChem.GetMorganFingerprintAsBitVect
        radius = int(args.radius)
        bits = int(args.bits)
        g1['bits'] = g1['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=radius, nBits=bits))
        ext['bits'] = ext['ROMol'].apply(lambda x: bit_generator(x, useChirality=True, radius=radius, nBits=bits))

        # vectorize
        vectorize = Grouping.Vector()
        logger.info('Generate ext vectors...')

        # divide ext set (each 10000 row)
        if len(ext) >= 10000:
            logger.info('Divide external set (over 10000 rows)')
            ext_vector = pd.DataFrame()
            for tmp in np.array_split(ext, (len(ext) // 10000) + 1):
                tmp_vector = vectorize.run(g1, tmp, n_cores)
                ext_vector = pd.concat([ext_vector, tmp_vector])
        else:
            ext_vector = vectorize.run(g1, ext, n_cores)

        # scaling
        logger.info('Scaling vectors...')
        scaler = load(str(path_scaler))
        cols = [col for col in ext_vector.columns if col.startswith('f_')]
        ext_vector[cols] = scaler.transform(ext_vector[cols])

        # save vector
        Utils.save(ext_vector, path_output / file_name, custom=f"ext_vector")
    else:
        ext = pd.read_csv(path_ext, sep='\t')
        ext_vector = ext.copy()

    # scoring
    if 'AD' in ext_vector.columns:
        ext_score = pd.DataFrame([result_scoring(model, ext_vector, args.name, path_output)])
    else:
        ext_score = predict_ext(model, ext_vector)
        logger.info(ext_score['Pred'].value_counts())

    # save
    Utils.save(ext_score, path_output / file_name, custom=f'{args.name}_predict_log')

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")


