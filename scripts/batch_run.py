#!/usr/bin/env python
# coding: utf-8
"""Run combinations of models and dataset in batch"""

import json
import pickle
from pathlib import Path

import pandas as pd
from mlxtend.classifier import StackingClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier


def main():

    # utility functions
    def select_channels(data_df, channels):
        """Select a list of channels"""
        return data_df.loc[(slice(None), slice(None), channels), :].unstack('channel')

    def prep_data(dataframe):
        """Prepare data from dataframe"""
        X = dataframe.sample(frac=1)
        y = dataframe.reset_index('healthy')['healthy']  # index 'healthy' is used as the class label
        y = y ^ 1  # invert y labels
        return X, y

    def run_datasets(datasets, pipe_template, param_grids):
        """
        Batch run combinations
        """
        results = list()
        test_results = list()
        for ds_desc, ds in datasets:
            print('Fitting model on dataset {}'.format(ds_desc))
            X, y = prep_data(ds)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                stratify=y,
                random_state=SEED
            )
            cls = Pipeline(pipe_template)
            cv = list(StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED).split(X_train, y_train))
            grid = GridSearchCV(
                cls,
                param_grid=param_grids,
                n_jobs=1,
                cv=cv,
                refit=REFIT,
                return_train_score=True,
                scoring=SCORING
            )
            grid.fit(X_train, y_train)
            cv_results = grid.cv_results_
            runs = len(cv_results['params'])
            for i in range(runs):
                results.append(
                    [ds_desc] +
                    [GLOBAL_PARAMS] +
                    [cv_results[f][i] for f in grid_fields]
                )
            y_pred = grid.best_estimator_.predict(X_test)
            test_results.append(
                [ds_desc] +
                [GLOBAL_PARAMS] +
                [grid.best_params_] +
                [accuracy_score(y_test, y_pred),
                 precision_score(y_test, y_pred),
                 recall_score(y_test, y_pred),
                 roc_auc_score(y_test, y_pred)
                 ]
            )
        return results, test_results

    # Global configuration
    SEED = 24
    CV = 5
    TEST_SIZE = 0.2
    center_channels = ['Cz', 'FCz', 'Fz']
    batch2_common_channels = 'CP1 CP2 Cz FC1 FC2 FCz Fp1 Fp2 Fz O1 O2 Oz Pz'.split(' ')
    SCORING = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'auc': 'roc_auc',
    }
    REFIT = 'accuracy'
    global_hyperparameters = {
        'seed': SEED,
        'CV': CV,
        'Test size': TEST_SIZE,
        'scoring': SCORING,
        'refit': REFIT
    }
    GLOBAL_PARAMS = json.dumps(global_hyperparameters)

    # Results configurations
    prefixes = ['_'.join([t1, t2]) for t1 in ['mean', 'std'] for t2 in ['train', 'test']]
    metric_names = ['_'.join([pre, met]) for pre in prefixes for met in SCORING.keys()]
    grid_fields = ['params'] + metric_names
    headers = ['Dataset', 'Global params', 'Parameters'] + metric_names
    test_headers = ['Dataset', 'Global params', 'Parameters', 'test_accuracy', 'test_precision', 'test_recall', 'test_auc']
    root = Path('/tmp/working/fang/end2end_run/baseline_ds_models_2')
    if not root.is_dir():
        root.mkdir(parents=True)
    results_file = root / 'cv_result.csv'
    test_results_file = root / 'test_result.csv'
    results = list()
    test_results = list()

    # Load intermediate data
    unioned_feature_df_file = Path('/tmp/working/fang/data/unioned_feature_df.pkl')
    unioned_feature_df = pickle.load(unioned_feature_df_file.open('rb'))
    unioned_df_file = Path('/tmp/working/fang/data/unioned_df.pkl')
    unioned_df = pickle.load(unioned_df_file.open('rb'))
    unioned_bands_feature_df_file = Path('/tmp/working/fang/data/bands_feature_df.pkl')
    unioned_bands_feature_df = pickle.load(unioned_bands_feature_df_file.open('rb'))
    raw_plus_feature_df = pd.concat([unioned_df, unioned_feature_df], keys=['raw', 'feature'], axis=1)
    raw_plus_all_feature_df = pd.concat([unioned_df, unioned_feature_df, unioned_bands_feature_df],
                                    keys=['raw', 'feature', 'bands feature'],
                                    axis=1)
    all_feature_df = pd.concat([unioned_feature_df, unioned_bands_feature_df], axis=1)

    # Construct datasets
    # feature datasets
    ds1 = select_channels(unioned_feature_df, center_channels)
    ds2, ds3, ds4 = (select_channels(unioned_feature_df, [ch]) for ch in center_channels)
    datasets = [
        ('Stats features, center channels', ds1),
        ('Stats features, Cz channel', ds2),
        ('Stats features, FCz channel', ds3),
        ('Stats features, Fz channel', ds4)
    ]

    nfds1 = select_channels(all_feature_df, center_channels)
    nfds2, nfds3, nfds4 = (select_channels(all_feature_df, [ch]) for ch in center_channels)
    nfdatasets = [
        ('stats, bands feature, center channels', nfds1),
        ('stats, bands feature, Cz channel', nfds2),
        ('stats, bands feature, FCz channel', nfds3),
        ('stats, bands feature, Fz channel', nfds4)
    ]

    bds1 = select_channels(raw_plus_feature_df, center_channels)
    bds2, bds3, bds4 = (select_channels(raw_plus_feature_df, [ch]) for ch in center_channels)
    bdatasets = [
        ('Raw plus stats feature, center channels', bds1),
        ('Raw plus stats feature, Cz channel', bds2),
        ('Raw plus stats feature, FCz channel', bds3),
        ('Raw plus stats feature, Fz channel', bds4)
    ]

    nds1 = select_channels(raw_plus_all_feature_df, center_channels)
    nds2, nds3, nds4 = (select_channels(raw_plus_all_feature_df, [ch]) for ch in center_channels)

    ndatasets = [
        ('Raw plus stats, bands feature, center channels', nds1),
        ('Raw plus stats, bands feature, Cz channel', nds2),
        ('Raw plus stats, bands feature, FCz channel', nds3),
        ('Raw plus stats, bands feature, Fz channel', nds4),
    ]

    # Pipeline configuration
    pipe_template1 = [
        ('cls', None)
    ]

    base_models = [
        LinearSVC(random_state=SEED),
        SVC(random_state=SEED),
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        AdaBoostClassifier(learning_rate=0.75, random_state=SEED),
        ExtraTreesClassifier(n_estimators=100, random_state=SEED),
        GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        XGBClassifier(n_estimators=100),
        KNeighborsClassifier(n_neighbors=5),
        GaussianNB()
    ]
    cls = StackingClassifier(
            classifiers=base_models,
            meta_classifier=LogisticRegression()
        )

    # Configuration without dimension reduction
    param_grids1 = [
        {
            'cls': [
                LinearSVC(random_state=SEED),
                SVC(random_state=SEED),
                RandomForestClassifier(n_estimators=100, random_state=SEED),
                AdaBoostClassifier(learning_rate=0.75, random_state=SEED),
                ExtraTreesClassifier(n_estimators=100, random_state=SEED),
                GradientBoostingClassifier(n_estimators=100, random_state=SEED),
                XGBClassifier(n_estimators=100),
                KNeighborsClassifier(n_neighbors=5),
                GaussianNB(),
                cls
            ]
        }
    ]

    small_ds = datasets + nfdatasets
    large_ds = bdatasets + ndatasets

    print('Running small datasets without dimension reduction...')
    res, test_res = run_datasets(small_ds, pipe_template1, param_grids1)
    results.extend(res)
    test_results.extend(test_res)

    # Configurations with dimension reduction
    pipe_template2 = [
        ('dim_red', None),
        ('cls', None)
    ]
    param_grids2 = [
        {
            'dim_red': [None, PCA(n_components=100)],
            'cls': [
                LinearSVC(random_state=SEED),
                SVC(random_state=SEED),
                RandomForestClassifier(n_estimators=100, random_state=SEED),
                AdaBoostClassifier(learning_rate=0.75, random_state=SEED),
                ExtraTreesClassifier(n_estimators=100, random_state=SEED),
                GradientBoostingClassifier(n_estimators=100, random_state=SEED),
                XGBClassifier(n_estimators=100),
                KNeighborsClassifier(n_neighbors=5),
                GaussianNB(),
                cls
            ]
        }
    ]

    print('Running large datasets with dimension reduction...')
    res, test_res = run_datasets(large_ds, pipe_template2, param_grids2)
    results.extend(res)
    test_results.extend(test_res)

    # Save results
    results_df = pd.DataFrame(results, columns=headers)
    results_df.to_csv(results_file.absolute(), index=False)
    test_results_df = pd.DataFrame(test_results, columns=test_headers)
    test_results_df.to_csv(test_results_file.absolute(), index=False)


if __name__ == '__main__':
    main()