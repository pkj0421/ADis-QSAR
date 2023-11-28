import time
import logging
import warnings
import argparse
import pandas as pd
import xgboost as xgb

from joblib import dump
from pathlib import Path
from Admodule import Utils

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, precision_score, confusion_matrix, r2_score

"""
This code is used for model training
Need train and test vectors

The model algorithms used SVM, RandomForest, XGBoost
The tuning parameters used for each algorithm can be freely changed in the code below

Optimizer uses GridSearchCV
Models are created with precision as the top priority

The output is trained model(.pkl) and log files(.tsv) files
"""

# module info option
logger = logging.getLogger(__name__)


def GridSearchRUN(model, parameters, X_train, y_train, cores=4):
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, zero_division=0),
               'auc': make_scorer(roc_auc_score)}
    grid_model = GridSearchCV(model, param_grid=parameters,
                              scoring=scoring, refit='precision',
                              cv=10, n_jobs=cores, error_score=0,
                              verbose=10)
    grid_model.fit(X_train, y_train)
    result = pd.DataFrame(grid_model.cv_results_['params'])
    result['mean_valid_accuracy'] = grid_model.cv_results_['mean_test_accuracy']
    result['mean_valid_precision'] = grid_model.cv_results_['mean_test_precision']
    result['mean_valid_auc'] = grid_model.cv_results_['mean_test_auc']
    result.sort_values(by='mean_valid_precision', ascending=False, inplace=True)
    logger.info(f"GridSearchCV results :\n{result[result.columns[-3:]]}")
    return grid_model, result


def confusion_matrix_scorer(y, y_pred, y_proba):
    cm = confusion_matrix(y, y_pred)
    auc = round(roc_auc_score(y, y_proba), 2)
    r2 = round(r2_score(y, y_pred), 2)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'auc': auc, 'r2': r2}


def result_scoring(md, df, cols, name, out):
    # divide X, y
    X = df[cols]
    y = df['AD']

    # make dataframe
    y_pred = md.predict(X)
    y_proba = md.predict_proba(X)[:, 1]
    df['Pred'] = y_pred
    df['Match'] = df[['AD', 'Pred']].apply(lambda x: 1 if x[0] == x[1] else 0, axis=1)
    Utils.save(df[['Compound_ID', 'AD', 'Pred', 'Match']], out, custom=f"{name}_prediction_log")

    # confusion matrix
    cm = confusion_matrix_scorer(y, y_pred, y_proba)
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
        accuracy = round(answer / len(y), 2)
    else:
        accuracy = 0

    result = {'Data': name,
              'ACC': f"{accuracy} ({answer} / {len(y)})",
              'TP': cm['tp'], 'FP': cm['fp'], 'FN': cm['fn'], 'TN': cm['tn'],
              'Precision': f"{precision} ({cm['tp']} / {cm['tp']} + {cm['fp']})",
              'Recall': f"{recall} ({cm['tp']} / {cm['tp']} + {cm['fn']})",
              'F1': f"{f1} (2 * ({recall} * {precision}) / {recall} + {precision})",
              'Specificity': f"{specificity} ({cm['tn']} / {cm['tn']} + {cm['fp']})",
              'AUC': cm['auc'], 'r2': cm['r2']}

    logger.info(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ADis QSAR model')
    parser.add_argument('-train', '--train', required=True, help='Train data')
    parser.add_argument('-valid', '--valid', required=True, help='Valid data')
    parser.add_argument('-test', '--test', default=False, help='Test data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')
    parser.add_argument('-m', '--model', type=str, default='RF', help='Set your model type')
    parser.add_argument('-core', '--num_cores', type=int, default=2, help='Set the number of CPU cores to use')
    args = parser.parse_args()

    # ignore warnings
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    # path
    path_train = Path(args.train)
    path_valid = Path(args.valid)
    file_name = path_train.stem.split('_')[0]
    path_output = Path(args.output) / f"{file_name}_model" / args.model
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, 'model.log')

    # Start
    start = time.time()

    # set cores
    n_cores = Utils.set_cores(args.num_cores)

    # initial info
    logger.info(f'Train data : {path_train}')
    logger.info(f'Valid data : {path_valid}')
    logger.info(f'Output path : {path_output}')
    logger.info(f'Model type : {args.model}')
    logger.info(f'Use cores : {n_cores}')

    # load data
    train = pd.read_csv(path_train, sep='\t')
    valid = pd.read_csv(path_valid, sep='\t')
    xcols = [x for x in train.columns if x.startswith('f_')]

    # data info
    logger.info('Start Learning model')
    logger.info(f"Train : {len(train)} | Valid : {len(valid)}")

    # model init
    if args.model.upper() == 'SVM':
        model = SVC(random_state=42)
        parameters = {"kernel": ['linear', 'rbf'],
                      "C": [0.01, 0.1, 1, 10, 100, 1000],
                      "gamma": [0.01, 0.1, 1, 100, 1000],
                      }

    elif args.model.upper() == 'RF':
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
        parameters = {"max_depth": [6, 8, 10, 12],
                      "max_features": ['sqrt', 'log2'],
                      "min_samples_leaf": [2, 4, 8, 10],
                      "min_samples_split": [2, 4, 8, 10],
                      "n_estimators": [200, 600, 1000],
                      }

    elif args.model.upper() == 'XGB':
        pos_ratio = round(len(train[train['AD'] == 0]) / len(train[train['AD'] == 1]), 2)
        model = xgb.XGBClassifier(scale_pos_weight=pos_ratio, n_jobs=0, seed=42)
        parameters = {"max_depth": [4, 6, 8],
                      "learning_rate": [0.01, 0.2],
                      "n_estimators": [200, 600, 1000],
                      "gamma": [0, 1000],
                      "min_child_weight": [1, 3],
                      "subsample": [0.5, 1],
                      "colsample_bytree": [0.5, 1],
                      "objective" : ['binary:logitraw'],
                      "eval_metric": ['auc', 'error'],
                      }

    elif args.model.upper() == 'MLP':
        model = MLPClassifier(random_state=42)
        parameters = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      'activation': ['relu', 'identity', 'logistic', 'tanh'],
                      'solver': ['adam', 'sgd', 'lbfgs'],
                      'alpha': [0.1, 0.01, 0.001],
                      'max_iter': [500, 1000],
                      'early_stopping': [True]
                      }

    else:
        logger.error(f"{args.model} model type can not use.")

    logger.info(f"Set parameters : {parameters}")

    # start learning
    grid, train_log = GridSearchRUN(model, parameters, train[xcols], train['AD'], cores=n_cores)
    grid_model = grid.best_estimator_

    logger.info(f"Best model :\n{grid_model}")

    # scoring
    train_score = result_scoring(grid_model, train, xcols, 'train', out=path_output / f"{file_name}_{args.model}")
    valid_score = result_scoring(grid_model, valid, xcols, 'valid', out=path_output / f"{file_name}_{args.model}")
    if args.test:  # if you have
        path_test = Path(args.test)
        test = pd.read_csv(path_test, sep='\t')
        test_score = result_scoring(grid_model, test, xcols, 'test', out=path_output / f"{file_name}_{args.model}")
        total_score = pd.DataFrame([train_score, valid_score, test_score])
    else:
        total_score = pd.DataFrame([train_score, valid_score])

    # save model
    dump(grid_model, path_output / f'{file_name}_{args.model}_model.pkl')
    Utils.save(train_log, path_output / f"{file_name}_{args.model}_model", custom='training_log')
    Utils.save(total_score, path_output / f"{file_name}_{args.model}_model", custom='score_log')

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")


