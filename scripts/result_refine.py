#!/usr/bin/env python
# coding: utf-8

# Module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from collections import OrderedDict
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import skew, kurtosis
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier


def parse_param(param):
    """
    To be applied to pandas Series objects
    :return: a series of 'classifier name', 'dim reducer', 'dim reduced to', 'classifier params'
    """
    result = dict()
    param_dict = eval(param)
    result['classifier name'] = param_dict['cls'].__class__.__name__
    if 'dim_red' not in param_dict or param_dict['dim_red'] is None:
        result['dim reducer'] = 'None'
        result['dim reduced to'] = -1
    else:
        result['dim reducer'] = param_dict['dim_red'].__class__.__name__
        if 'dim reduced to' in param_dict:
            result['dim reduced to'] = param_dict['dim_red__n_components']
        else:
            result['dim reduced to'] = 100  # a magic number, can be improved
    cls_params = {name[5:]:param_dict[name] for name in param_dict.keys() if '__' in name and name.split('__')[0] == 'cls'}
    if not cls_params:
        result['classifier params'] = 'None'
    else:
        result['classifier params'] = str(cls_params)
    return pd.Series(result)


def parse_global(global_param):
    """
    To be applied to pandas Series objects
    :return: a series of 'CV folds', 'random seed', 'test size'
    """
    global_param = global_param.replace('""', '"')
    result = dict()
    param_dict = json.loads(global_param)
    result['CV folds'] = int(param_dict['CV'])
    result['random seed'] = int(param_dict['seed'])
    result['test size'] = param_dict['Test size']
    return pd.Series(result)

def parse_dataset(ds_desc):
    """
    To be applied to pandas Series objects
    :return: a series of 'stats features', 'bands features', 'channels', 'raw'
    """
    result = dict()
    result['raw'] = 'no'
    if ds_desc.startswith('Features only'):
        result['stats features'] = 'yes'
        result['bands features'] = 'no'
    elif 'stats, bands' in ds_desc:
        result['stats features'] = 'yes'
        result['bands features'] = 'yes'
    elif 'Raw plus feature' in ds_desc:
        result['stats features'] = 'yes'
        result['bands features'] = 'no'
    else:
        result['stats features'] = 'no'
        result['bands features'] = 'no'

    if ds_desc.startswith('Raw plus'):
        result['raw'] = 'Yes'
    else:
        result['raw'] = 'no'

    if ds_desc.endswith('center channels'):
        result['channels'] = 'all'
    else:
        result['channels'] = ds_desc.split(',')[-1].strip().split(' ')[-2]

    assert len(result.keys()) == 4
    assert result['channels'] in ['Fz', 'FCz', 'Cz', 'all'], result['channels']

    return pd.Series(result)

def main():
    root = Path('/tmp/working/fang/end2end_run/')
    files = root.glob('*/*.csv')

    root = Path('/tmp/working/fang/end2end_run/')
    for f in root.glob('*/*.csv'):
        if f.parent.name == 'baseline_ds_models_2':
            continue
        if f.name.endswith('_parsed.csv'):
            continue
        print(f)
        new_path = f.parent / (str(f.stem) + '_parsed.csv')
        # if new_path.is_file():
        #     print('Skipping parsed result: {}'.format(f.name))
        #     continue  # skipping parsed
        df = pd.read_csv(f.absolute())
        global_df = df['Global params'].apply(parse_global)
        param_df = df['Parameters'].apply(parse_param)
        ds_df = df['Dataset'].apply(parse_dataset)
        result_df = pd.concat(
            [
            df.drop(['Global params', 'Parameters', 'Dataset'], axis=1),
            global_df,
            param_df,
            ds_df
            ],
            axis=1)
        result_df.to_csv(new_path.absolute(), index=False)


if __name__ == '__main__':
    main()