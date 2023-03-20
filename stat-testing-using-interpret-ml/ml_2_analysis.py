# -*- coding: utf-8 -*-
'''
Statistical testing using interpretable machine-learning
v662
@author: Dr. David Steyrl david.steyrl@gmail.com
'''

import numpy as np
import os
import pandas as pd
import pickle
import shutil
import warnings
from collections import Counter
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from shap.explainers import Exact as ExactExplainer
from shap.explainers import Linear as LinearExplainer
from shap.explainers import Partition as PartitionExplainer
from shap.explainers import Tree as TreeExplainer
from shap.maskers import Impute as ImputeMasker
from shap.maskers import Independent as IndependentMasker
from shap.maskers import Partition as PartitionMasker
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from time import time

# Create warning string
warnings_string = ('ignore:X does not have valid feature names:::')
# Ignore warning string
os.environ['PYTHONWARNINGS'] = warnings_string
# Filter out specific warnings
warnings.filterwarnings('ignore', 'X does not have valid feature names')


def create_dir(path):
    '''
    Create specified directory if not existing.

    Parameters
    ----------
    path : string
        Path to to check to be created.

    Returns
    -------
    None.

    '''
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)


def drop_nan_rows(g, x, y):
    '''
    Identify and drop rows containing nans from dataframes.

    Parameters
    ----------
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Targets dataframe.

    Returns
    -------
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Targets dataframe.

    '''
    # Search for nans in predictors
    rows_nans = list(x.loc[x.isna().any(axis=1).to_numpy(), :].index)
    # Drop rows from cv groups
    g = g.drop(rows_nans).reset_index(drop=True)
    # Drop rows from predictors
    x = x.drop(rows_nans).reset_index(drop=True)
    # Drop rows from targets
    y = y.drop(rows_nans).reset_index(drop=True)
    # Return results
    return g, x, y


def prepare(task):
    '''
    Prepare analysis pipeline, prepare seach_space.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.

    Returns
    -------
    pipe : scikit-learn compatible analysis pipeline
        Prepared pipe object.
    space : dict
        Space that should be searched for optimale parameters.

    '''
    # Pipeline for continous data preprocessing -------------------------------
    # One hot encoder
    ohe_1 = OneHotEncoder(categories=task['ohe_categories'],
                          drop='if_binary',
                          sparse_output=False,
                          dtype=np.float64,
                          handle_unknown='error',
                          min_frequency=None,
                          max_categories=None)
    # OHE categorical predictors
    coltrans_1 = ColumnTransformer(
        [('ohe_1', ohe_1, task['X_CAT_NAMES'])],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Standard scaler
    stdscal_1 = StandardScaler(copy=True,
                               with_mean=True,
                               with_std=True)
    # Imputer
    knnimp_1 = KNNImputer(missing_values=np.nan,
                          n_neighbors=3,
                          weights='distance',
                          metric='nan_euclidean',
                          copy=True,
                          add_indicator=False,
                          keep_empty_features=False)
    # Column selector
    coltrans_2 = ColumnTransformer(
        [('pass', 'passthrough', task['X_CON_NAMES'])],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Pipeline
    con_pipe = Pipeline([('ohe_cat', coltrans_1),
                         ('std_scaler', stdscal_1),
                         ('knn_impute', knnimp_1),
                         ('con_selector', coltrans_2)],
                        memory=None,
                        verbose=False)

    # Pipeline for categorical data preprocessing -----------------------------
    # Categorical data one hot encoder
    ohe_2 = OneHotEncoder(categories=task['ohe_categories'],
                          drop='if_binary',
                          sparse_output=False,
                          dtype=np.float64,
                          handle_unknown='error',
                          min_frequency=None,
                          max_categories=None)
    # OHE categorical predictors
    coltrans_3 = ColumnTransformer(
        [('ohe_2', ohe_2, task['X_CAT_NAMES'])],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Standard scaler
    stdscal_2 = StandardScaler(copy=True,
                               with_mean=True,
                               with_std=True)
    # Pipeline
    cat_pipe = Pipeline([('ohe_cat', coltrans_3),
                         ('std_scaler', stdscal_2)],
                        memory=None,
                        verbose=False)

    # Preprocessing pipeline --------------------------------------------------
    # Put pipes together
    pre_pipe = ColumnTransformer(
        [('con', con_pipe, task['X_CON_NAMES']+task['X_CAT_NAMES']),
         ('cat', cat_pipe, task['X_CAT_NAMES'])],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)

    # Regression --------------------------------------------------------------
    if task['KIND'] == 'reg':
        # Linear model
        if task['ESTIMATOR_NAME'] == 'LM':
            # Estimator
            estimator = ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                fit_intercept=True,
                precompute=False,
                max_iter=1000,
                copy_X=True,
                tol=0.001,
                warm_start=True,
                positive=False,
                random_state=None,
                selection='cyclic')
            # Search space
            space = {
                'estimator__regressor__alpha': loguniform(0.0001, 1),
                'estimator__regressor__l1_ratio': uniform(0.01, 0.99),
                }
        # Gradient boosting model
        elif task['ESTIMATOR_NAME'] == 'GB':
            # Estimator
            estimator = LGBMRegressor(
                boosting_type='goss',
                num_leaves=200,
                max_depth=-1,
                learning_rate=0.1,
                n_estimators=100,
                subsample_for_bin=100000,
                objective='regression_l2',
                class_weight=None,
                min_split_gain=0.0,
                min_child_weight=0.001,
                min_child_samples=2,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1.0,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=None,
                n_jobs=1,
                importance_type='gain',
                **{'top_rate': 0.5,
                   'feature_pre_filter': False,
                   'max_bin': 1000,
                   'min_data_in_bin': 1})
            # Search space
            space = {
                'estimator__regressor__learning_rate': loguniform(0.01, 0.1),
                'estimator__regressor__n_estimators': randint(100, 1001),
                'estimator__regressor__colsample_bytree': uniform(0.5, 0.5),
                'estimator__regressor__path_smooth': uniform(0, 1001),
                'estimator__regressor__extra_trees': [True, False],
                }
        # Other
        else:
            # Raise error
            raise TypeError('Estimator name not found.')
        # Add scaler to the estimator
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            func=None,
            inverse_func=None,
            check_inverse=True)

    # Classification ----------------------------------------------------------
    elif task['KIND'] == 'clf':
        # Linear model
        if task['ESTIMATOR_NAME'] == 'LM':
            # Estimator
            estimator = LogisticRegression(
                penalty='elasticnet',
                dual=False,
                tol=0.001,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=1,
                class_weight=None,
                random_state=None,
                solver='saga',
                max_iter=1000,
                multi_class='multinomial',
                verbose=0,
                warm_start=True,
                n_jobs=None,
                l1_ratio=None)
            # Search space
            space = {
                'estimator__C': loguniform(1, 10000),
                'estimator__l1_ratio': uniform(0.01, 0.99),
                }
        # Gradient boosting model
        elif task['ESTIMATOR_NAME'] == 'GB':
            # Estimator
            estimator = LGBMClassifier(
                boosting_type='goss',
                num_leaves=200,
                max_depth=-1,
                learning_rate=0.1,
                n_estimators=100,
                subsample_for_bin=100000,
                objective='cross_entropy',
                class_weight=None,
                min_split_gain=0.0,
                min_child_weight=0.001,
                min_child_samples=2,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1.0,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=None,
                n_jobs=1,
                importance_type='gain',
                **{'top_rate': 0.5,
                   'feature_pre_filter': False,
                   'max_bin': 1000,
                   'min_data_in_bin': 1})
            # Search space
            space = {
                'estimator__learning_rate': loguniform(0.01, 0.1),
                'estimator__n_estimators': randint(100, 1001),
                'estimator__colsample_bytree': uniform(0.5, 0.5),
                'estimator__path_smooth': uniform(0, 1001),
                'estimator__extra_trees': [True, False],
                }
        # Other
        else:
            # Raise error
            raise TypeError('Estimator name not found.')
    # Other
    else:
        # Raise error
        raise TypeError('Kind not found.')

    # Make pipeline -----------------------------------------------------------
    # Analyis pipeline
    pipe = Pipeline(
        [('preprocessing', pre_pipe),
         ('estimator', estimator)],
        memory=None,
        verbose=False).set_output(transform='pandas')
    # Return the pipe and space
    return pipe, space


def split_data(df, i_trn, i_tst):
    '''
    Split dataframe in training and testing dataframes.

    Parameters
    ----------
    df : dataframe
        Dataframe holding the data to split.
    i_trn : numpy array
        Array with indices of training data.
    i_tst : numpy array
        Array with indices of testing data.

    Returns
    -------
    df_trn : dataframe
        Dataframe holding the training data.
    df_tst : dataframe
         Dataframe holding the testing data.

    '''
    # Dataframe is not empty
    if not df.empty:
        # Make split
        df_trn = df.iloc[i_trn].reset_index(drop=True)
        # Make split
        df_tst = df.iloc[i_tst].reset_index(drop=True)
    # Dataframe is empty
    else:
        # Make empty dataframes
        df_trn, df_tst = pd.DataFrame(), pd.DataFrame()
    # Return train test dataframes
    return df_trn, df_tst


def get_class_w(y):
    '''
    Compute class weights over array by counting occurrences.

    Parameters
    ----------
    y : ndarray
        Array containing class labels.

    Returns
    -------
    class_weights : dictionary
        Dictionary of class weights with class labels as keys.

    '''
    # Count unique classes occurances
    counter = Counter(y.squeeze())
    # n_samples
    total_class = sum(counter.values())
    # Get weights
    w = {key: np.round(count/total_class, 4) for key, count in counter.items()}
    # Return class weights
    return w


def weighted_accuracy_score(y_true, y_pred, class_weights):
    '''
    Computes accuracy score weighted by the inverse if the frequency of a
    class.

    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
    class_weights : dictionary
        Class weights as inverse of frequency of class.

    Returns
    -------
    accuracy : float
        Prediction accuracy.

    '''
    # Make sample weights dataframe
    w = y_true.squeeze().map(class_weights).to_numpy()
    # Return sample weighted accuracy
    return accuracy_score(y_true, y_pred, sample_weight=w)


def print_tune_summary(task, i_cv, n_splits, hp_params, hp_score):
    '''
    Print best paramters and related score to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current cv repetition.
    n_splits : int
        Number of splits in inner cv
    hp_params : dictionary
        Best hyper params found.
    hp_score : dictionary
        Score for best hyper params found.

    Returns
    -------
    None.

    '''
    # Regression
    if task['KIND'] == 'reg':
        # Print data set
        print('Dataset: '+task['PATH_TO_DATA'])
        # Print general information
        print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
              'n rep outer cv: '+str(task['N_REP_OUTER_CV'])+' | ' +
              'n rep inner cv: '+str(n_splits)+' | ' +
              str(task['ESTIMATOR_NAME'])+' | ' +
              'best R2: '+str(np.round(hp_score, decimals=4)))
        # Print best hyperparameter and related score for regression task
        print(str(hp_params))
    # Classification
    elif task['KIND'] == 'clf':
        # Print data set
        print('Dataset: '+task['PATH_TO_DATA'])
        # Print general information
        print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
              'n rep outer cv: '+str(task['N_REP_OUTER_CV'])+' | ' +
              'n rep inner cv: '+str(n_splits)+' | ' +
              str(task['ESTIMATOR_NAME'])+' | ' +
              'acc: '+str(np.round(hp_score, decimals=4)))
        # Print best hyperparameter and related score for classification task
        print(str(hp_params))
    # Other
    else:
        # Raise error
        raise TypeError('Kind not found.')


def tune_pipe(task, i_cv, pipe, space, g_trn, x_trn, y_trn):
    '''
    Inner loop of the nested cross-validation. Runs a search for optimal
    hyperparameter (random search).
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009.
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current iteration of outer cross-validation.
    pipe : pipeline object
        Analysis pipeline.
    space : dict
        Space that should be searched for optimale parameters.
    g_trn : ndarray (n_samples)
        Group data.
    x_trn : ndarray (n_features x n_samples)
        Predictor train data.
    y_trn : ndarray (n_samples)
        Target train data.

    Returns
    -------
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    best parameters : dict
        Best hyperparameters of the pipe.

    '''
    # Regression use R²
    if task['KIND'] == 'reg':
        # R2 score for regression
        scorer = 'r2'
    # Classification use weighted accuracy
    elif task['KIND'] == 'clf':
        # Weighted accuracy for classification
        scorer = make_scorer(weighted_accuracy_score,
                             greater_is_better=True,
                             **{'class_weights': get_class_w(y_trn)})
        # Add current class weights to the pipe
        pipe.set_params(**{'estimator__class_weight': get_class_w(y_trn)})
    # Other
    else:
        # Raise error
        raise TypeError('Kind not found.')

    # Tune analysis pipeline --------------------------------------------------
    # Choose n_splits to approx N_SAMPLES_INNER_CV predictions, min 5, max 1000
    n_splits = min(1000, max(5, int(task['N_SAMPLES_INNER_CV'] /
                                    (g_trn.shape[0]*task['TST_SIZE_FRAC']))))
    # Instanciate random parameter search
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=task['N_SAMPLES_RS'],
        scoring=scorer,
        n_jobs=task['N_JOBS'],
        refit=True,
        cv=GroupShuffleSplit(n_splits=n_splits,
                             test_size=task['TST_SIZE_FRAC']),
        verbose=0,
        pre_dispatch='2*n_jobs',
        random_state=None,
        error_score=0,
        return_train_score=False)
    # Random search for best parameter
    search.fit(x_trn, y_trn.squeeze(), groups=g_trn)
    # Print tune summary
    print_tune_summary(task, i_cv, n_splits, search.best_params_,
                       search.best_score_)
    # Return tuned analysis pipe
    return (search.best_estimator_, search.best_params_)


def score_predictions(task, pipe, x_tst, y_tst, y):
    '''
    Compute scores for predictions based on task.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Analysis pipeline.
    x_tst : ndarray (n_features x n_samples)
        Predictor test data.
    y_tst : ndarray (n_samples)
        Target test data.
    y : ndarray
        All available target data to compute true class weights for scoring.

    Returns
    -------
    scores : dict
        Returns scoring results. MAE, MSE and R² if task is regression.
        ACC and true class weights if task is classification.

    '''
    # Predict test samples
    y_pred = pipe.predict(x_tst)
    # Regression
    if task['KIND'] == 'reg':
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of mse
        mse = mean_squared_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {'y_true': y_tst.squeeze().to_numpy(),
                  'y_pred': y_pred,
                  'mae': mae,
                  'mse': mse,
                  'r2': r2}
    # Classification
    elif task['KIND'] == 'clf':
        # Get class weights
        class_weights = get_class_w(y)
        # Calculate model fit in terms of acc
        acc = weighted_accuracy_score(y_tst, y_pred, class_weights)
        # Results
        scores = {'y_true': y_tst.squeeze().to_numpy(),
                  'y_pred': y_pred,
                  'acc': acc,
                  'class_weights': class_weights}
    # Other
    else:
        # Raise error
        raise TypeError('Kind not found.')
    # Return scores dictionary
    return scores


def get_explainations(task, pipe, x_trn, x_tst):
    '''
    Get SHAP (SHapley Additive exPlainations) model explainations.
    Ref: Molnar, Christoph. 'Interpretable machine learning. A Guide for
    Making Black Box Models Explainable', 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Lundberg, Scott M., and Su-In Lee. “A unified approach to
    interpreting model predictions.” Advances in Neural Information Processing
    Systems. 2017.
    Ref: Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. “Consistent
    individualized feature attribution for tree ensembles.” arXiv preprint
    arXiv:1802.03888 (2018).
    Ref: Sundararajan, Mukund, and Amir Najmi. “The many Shapley values for
    model explanation.” arXiv preprint arXiv:1908.08474 (2019).
    Ref: Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. “Feature
    relevance quantification in explainable AI: A causality problem.” arXiv
    preprint arXiv:1910.13413 (2019).
    Ref: Slack, Dylan, et al. “Fooling lime and shap: Adversarial attacks on
    post hoc explanation methods.” Proceedings of the AAAI/ACM Conference on
    AI, Ethics, and Society. 2020.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    x_trn : ndarray (n_features x n_samples)
        Background data.
    x_tst : ndarray (n_features x n_samples)
        Test data for shap computation.

    Returns
    -------
    imp : shap explainer object
        SHAP based predictor importance.

    '''
    # Linear models and interactions
    if (task['ESTIMATOR_NAME'] == 'LM' and task['SHAP_INTERACTIONS']):
        # Linear models cannot have interactions
        task['SHAP_INTERACTIONS'] = False
        # Warning
        warnings.warn('Linear models cannot have interactions.' +
                      'Computation skipped.', UserWarning)
    # Subsample train data to be used as background data
    x_background = x_trn.sample(
        n=min(x_trn.shape[0], task['MAX_SAMPLES_SHAP_BACKGROUND']),
        random_state=3141592,
        ignore_index=True)
    # Transform background data
    x_background = pipe[0].transform(x_background)
    # Subsample test data
    x_tst_shap_orig = x_tst.sample(
        n=min(x_tst.shape[0], task['MAX_SAMPLES_SHAP']),
        random_state=3141592,
        ignore_index=True)
    # Transform shap test data
    x_tst_shap = pipe[0].transform(x_tst_shap_orig)

    # Masker ------------------------------------------------------------------
    # Linear model and regression
    if (task['ESTIMATOR_NAME'] == 'LM' and task['KIND'] == 'reg'):
        # Get masker
        masker = ImputeMasker(
            x_background,
            method='linear')
    # Without interactions
    elif not task['SHAP_INTERACTIONS']:
        # Get masker
        masker = PartitionMasker(
            x_background,
            max_samples=task['MAX_SAMPLES_SHAP_BACKGROUND'],
            clustering='correlation')
    # With interactions
    else:
        # Get masker
        masker = IndependentMasker(
            x_background,
            max_samples=task['MAX_SAMPLES_SHAP_BACKGROUND'])

    # Explainer ---------------------------------------------------------------
    # Gradient boosting model and regression
    if (task['ESTIMATOR_NAME'] == 'GB' and task['KIND'] == 'reg'):
        # Get explainer
        explainer = TreeExplainer(
            pipe[1].regressor_,
            data=None,
            model_output='raw',
            feature_perturbation='tree_path_dependent',
            feature_names=None,
            approximate=False)
    # Linear model and regression
    elif (task['ESTIMATOR_NAME'] == 'LM' and task['KIND'] == 'reg'):
        # Get explainer
        explainer = LinearExplainer(
            pipe[1].regressor_,
            masker,
            nsamples=task['MAX_SAMPLES_SHAP_BACKGROUND'])
    # Without interactions
    elif not task['SHAP_INTERACTIONS']:
        # Get explainer
        explainer = PartitionExplainer(pipe[1].predict, masker)
    # With interactions
    else:
        # Get explainer
        explainer = ExactExplainer(pipe[1].predict, masker)

    # Explainations -----------------------------------------------------------
    # With interactions
    if task['SHAP_INTERACTIONS']:
        # Gradient boosting model and regression
        if (task['ESTIMATOR_NAME'] == 'GB' and task['KIND'] == 'reg'):
            # Get shap values
            shap_explainations = explainer(x_tst_shap,
                                           interactions=True,
                                           check_additivity=False)
        # Other
        else:
            # Get shap values
            shap_explainations = explainer(x_tst_shap,
                                           max_evals='auto',
                                           main_effects=False,
                                           error_bounds=False,
                                           batch_size='auto',
                                           interactions=True,
                                           silent=False)
    # Without interactions
    elif not task['SHAP_INTERACTIONS']:
        # Gradient boosting model and regression
        if (task['ESTIMATOR_NAME'] == 'GB' and task['KIND'] == 'reg'):
            # Get shap values
            shap_explainations = explainer(x_tst_shap,
                                           interactions=False,
                                           check_additivity=False)
        # Linear model and regression
        elif (task['ESTIMATOR_NAME'] == 'LM' and task['KIND'] == 'reg'):
            # Get shap values
            shap_explainations = explainer(x_tst_shap,
                                           max_evals='auto',
                                           main_effects=False,
                                           error_bounds=False,
                                           batch_size='auto',
                                           outputs=None,
                                           silent=False)
        # Other
        else:
            # Get shap values
            shap_explainations = explainer(x_tst_shap,
                                           max_evals='auto',
                                           main_effects=False,
                                           error_bounds=False,
                                           batch_size='auto',
                                           outputs=None,
                                           silent=False)
    # Replace scaled data in shap explainations with unscaled
    shap_explainations.data = pipe[0].transformers_[0][1][0].transform(
        x_tst_shap_orig).reindex(columns=task['x_names'])
    # Regression
    if task['KIND'] == 'reg':
        # Rescale shap values from scaled data to original space
        shap_explainations.values = (shap_explainations.values *
                                     pipe[1].transformer_.scale_[0])
        # Rescale shap base values from scaled data to original space
        shap_explainations.base_values = ((shap_explainations.base_values *
                                          pipe[1].transformer_.scale_[0]) +
                                          pipe[1].transformer_.mean_[0])
    # Return shap explainations
    return shap_explainations


def s2p(path_save, variable):
    '''
    Save variable as pickle file at path.

    Parameters
    ----------
    path_save : string
        Path ro save variable.
    variable : string
        Variable to save.

    Returns
    -------
    None.

    '''
    # Save variable as pickle file
    with open(path_save, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(variable, filehandle)


def print_current_results(task, t_start, scores, scores_sh):
    '''
    Print current results to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    t_start : time
        Start time of the current cross-validation loop.
    scores : dict
        Scores dict.
    scores_sh : dict
        Scores with shuffled data dict.

    Returns
    -------
    None.

    '''
    # Regression
    if task['KIND'] == 'reg':
        # Print current R2
        print('Current CV loop R2: '+str(np.round(
            scores[-1]['r2'], decimals=4)))
        # Print running mean R2
        print('Running mean R2: '+str(np.round(
            np.mean([i['r2'] for i in scores]), decimals=4)))
        # Print running mean shuffle R2
        print('Running shuffle mean R2: '+str(np.round(
            np.mean([i['r2'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print('Elapsed time: '+str(np.round(
            time() - t_start, decimals=1)), end='\n\n')
    # Classification
    elif task['KIND'] == 'clf':
        # Print current acc
        print('Current CV loop acc: '+str(np.round(
            scores[-1]['acc'], decimals=4)))
        # Print running mean acc
        print('Running mean acc: '+str(np.round(
            np.mean([i['acc'] for i in scores]), decimals=4)))
        # Print running mean shuffle acc
        print('Running shuffle mean acc: '+str(np.round(
            np.mean([i['acc'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print('Elapsed time: '+str(np.round(
            time() - t_start, decimals=1)), end='\n\n')
    # Other
    else:
        # Raise error
        raise TypeError('Kind not found.')


def cross_validation(task, g, x, y):
    '''
    Performe cross-validation analysis. Saves results to pickle file in
    path_to_results directory.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.


    Returns
    -------
    None.

    '''
    # Initialize results lists ------------------------------------------------
    # Initialize best params list
    best_params = []
    # Initialize score list
    scores = []
    # Initialize SHAP based explainations list
    explainations = []
    # Initialize shuffle data score list
    scores_sh = []
    # Initialize shuffle data SHAP based explainations list
    explainations_sh = []
    # Get analysis pipeline and space
    pipe, space = prepare(task)

    # Main cross-validation loop ----------------------------------------------
    # Instanciate main cv splitter with fixed random state for comparison
    cv = GroupShuffleSplit(
        n_splits=task['N_REP_OUTER_CV'],
        test_size=task['TST_SIZE_FRAC'],
        random_state=3141592)
    # Loop over main (outer) cross validation splits
    for i_cv, (i_trn, i_tst) in enumerate(cv.split(g, groups=g)):
        # Save loop start time
        t_start = time()

        # Split data ----------------------------------------------------------
        # Split groups
        g_trn, g_tst = split_data(g, i_trn, i_tst)
        # Split targets
        y_trn, y_tst = split_data(y, i_trn, i_tst)
        # Split predictors
        x_trn, x_tst = split_data(x, i_trn, i_tst)

        # Tune and fit --------------------------------------------------------
        # Get optimized and fitted pipe
        pipe, params = tune_pipe(task, i_cv, pipe, space, g_trn, x_trn, y_trn)
        # Store best params
        best_params.append(params)

        # Analyze -------------------------------------------------------------
        # Score predictions
        scores.append(score_predictions(task, pipe, x_tst, y_tst, y))
        # SHAP explainations
        explainations.append(get_explainations(task, pipe, x_trn, x_tst))

        # Shuffle data analyze ------------------------------------------------
        # Clone pipe
        pipe_sh = clone(pipe)
        # Refit pipe with shuffled targets
        pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
        # Score predictions
        scores_sh.append(score_predictions(task, pipe_sh, x_tst, y_tst, y))
        # SHAP explainations
        explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

        # Compile and save intermediate results and task ----------------------
        # Create results
        results = {
            'best_params': best_params,
            'scores': scores,
            'explainations': explainations,
            'scores_sh': scores_sh,
            'explainations_sh': explainations_sh
            }
        # Make save path
        save_path = task['path_to_results']+'/'+task['y_name'][0]
        # Save results as pickle file
        s2p(save_path+'_results.pickle', results)
        # Save task as pickle file
        s2p(save_path+'_task.pickle', task)

        # Print current results -----------------------------------------------
        print_current_results(task, t_start, scores, scores_sh)


def train_test_split(task, g, x, y):
    '''
    Performe train-test split analysis. Saves results to pickle file in
    path_to_results directory.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.


    Returns
    -------
    None.

    '''
    # Initialize results lists ------------------------------------------------
    # Initialize best params list
    best_params = []
    # Initialize score list
    scores = []
    # Initialize SHAP based explainations list
    explainations = []
    # Initialize shuffle data score list
    scores_sh = []
    # Initialize shuffle data SHAP based explainations list
    explainations_sh = []
    # Get analysis pipeline and space
    pipe, space = prepare(task)
    # Save start time
    t_start = time()

    # Split data --------------------------------------------------------------
    # Get train data index
    i_trn = list(set(g.index).difference(set(task['TEST_SET_IND'])))
    # Get test data index
    i_tst = task['TEST_SET_IND']
    # Splitting groups
    g_trn, g_tst = split_data(g, i_trn, i_tst)
    # Splitting targets
    y_trn, y_tst = split_data(y, i_trn, i_tst)
    # Splitting predictors
    x_trn, x_tst = split_data(x, i_trn, i_tst)

    # Tune and fit ------------------------------------------------------------
    # Get optimized and fitted pipe
    pipe, params = tune_pipe(task, 0, pipe, space, g_trn, x_trn, y_trn)
    # Store best params
    best_params.append(params)

    # Analyze -----------------------------------------------------------------
    # Score predictions
    scores.append(score_predictions(task, pipe, x_tst, y_tst, y))
    # SHAP explainations
    explainations.append(get_explainations(task, pipe, x_trn, x_tst))

    # Shuffle data analyze ----------------------------------------------------
    # Clone pipe
    pipe_sh = clone(pipe)
    # Refit pipe with shuffled targets
    pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
    # Score predictions
    scores_sh.append(score_predictions(task, pipe_sh, x_tst, y_tst, y))
    # SHAP explainations
    explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

    # Compile and save intermediate results and task ----------------------
    # Create results
    results = {
        'best_params': best_params,
        'scores': scores,
        'explainations': explainations,
        'scores_sh': scores_sh,
        'explainations_sh': explainations_sh
        }
    # Make save path
    save_path = task['path_to_results']+'/'+task['y_name'][0]
    # Save results as pickle file
    s2p(save_path+'_results.pickle', results)
    # Save task as pickle file
    s2p(save_path+'_task.pickle', task)

    # Print current results -----------------------------------------------
    print_current_results(task, t_start, scores, scores_sh)


def main():
    '''
    Main function of the machine-learning based data analysis.

    Returns
    -------
    None.

    '''
    ###########################################################################
    # Specify analysis
    ###########################################################################

    # 1. Specify task ---------------------------------------------------------

    # Type of analysis. string
    # Repeated Cross-validation: CV
    # Single Train-Test split: TT
    TYPE = 'CV'
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    N_JOBS = -2
    # CV: Number of outer CV repetitions. int (default: 100)
    N_REP_OUTER_CV = 50
    # CV & TT: Total number of predictions in inner CV. int (default: 20000)
    N_SAMPLES_INNER_CV = 20000
    # CV & TT: Test size fraction of groups in CV. float (]0,1], default: 0.2)
    TST_SIZE_FRAC = 0.2
    # Number of samples in random search. int (default: 500)
    N_SAMPLES_RS = 500
    # Estimator. string (LM linear model, GB gradient boosting, default: GB)
    ESTIMATOR_NAME = 'GB'
    # Limit number of samples for SHAP. int (default: 10).
    MAX_SAMPLES_SHAP = 10
    # Limit number of samples for background data in SHAP. int (default: 100)
    MAX_SAMPLES_SHAP_BACKGROUND = 100
    # Get SHAP interactions. Warning! Time consuming! bool (default: False)
    SHAP_INTERACTIONS = False
    # Drop rows with nans. If false imputation & ohe of nans (default: False)
    DROP_NAN = False

    # 2. Specify data ---------------------------------------------------------

    # # Cancer data - classification, 2 classes
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'cancer'+'_'+TYPE
    # # Specify task KIND. string (clf, reg)
    # KIND = 'clf'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/cancer_20221123.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id']
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'mean_radius',
    #     'mean_texture',
    #     'mean_perimeter',
    #     'mean_area',
    #     'mean_smoothness',
    #     'mean_compactness',
    #     'mean_concavity',
    #     'mean_concave_points',
    #     'mean_symmetry',
    #     'mean_fractal_dimension',
    #     'radius_error',
    #     'texture_error',
    #     'perimeter_error',
    #     'area_error',
    #     'smoothness_error',
    #     'compactness_error',
    #     'concavity_error',
    #     'concave_points_error',
    #     'symmetry_error',
    #     'fractal_dimension_error',
    #     'worst_radius',
    #     'worst_texture',
    #     'worst_perimeter',
    #     'worst_area',
    #     'worst_smoothness',
    #     'worst_compactness',
    #     'worst_concavity',
    #     'worst_concave_points',
    #     'worst_symmetry',
    #     'worst_fractal_dimension']
    # # Specify categorical predictor names. list of string or []
    # X_CAT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'target']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 569, size=114))

    # # Diabetes data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'diabetes'+'_'+TYPE
    # # Specify task KIND. string (clf, reg)
    # KIND = 'reg'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/diabetes_20220824.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id']
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'age',
    #     'bmi',
    #     'bp',
    #     's1',
    #     's2',
    #     's3',
    #     's4',
    #     's5',
    #     's6']
    # # Specify categorical predictor names. list of string or []
    # X_CAT_NAMES = [
    #     'sex']
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'target']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 441, size=88))

    # # Housing data - regression, 20k samples, categorical predictor
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'housing'+'_'+TYPE
    # # Specify task KIND. string (clf, reg)
    # KIND = 'reg'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/housing_20220824.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id']
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'longitude',
    #     'latitude',
    #     'housing_median_age',
    #     'total_rooms',
    #     'total_bedrooms',
    #     'population',
    #     'households',
    #     'median_income']
    # # Specify categorical predictors names. list of string or []
    # X_CAT_NAMES = [
    #     'ocean_proximity']
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'median_house_value_k']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 20639, size=4128))

    # Radon data - regression, high cardial categorical predictors
    # Specifiy an analysis name
    ANALYSIS_NAME = 'radon'+'_'+TYPE
    # Specify task KIND. string (clf, reg)
    KIND = 'reg'
    # Specify path to data. string
    PATH_TO_DATA = 'data/radon_20220824.xlsx'
    # Specify sheet name. string
    SHEET_NAME = 'data'
    # Specify grouping for CV split. list of string
    G_NAME = [
        'sample_id']
    # Specify continous predictor names. list of string or []
    X_CON_NAMES = [
        'Uppm']
    # Specify categorical predictors names. list of string or []
    X_CAT_NAMES = [
        'county_code',
        'floor']
    # Specify target name(s). list of strings or []
    Y_NAMES = [
        'log_radon']
    # Rows to skip. list of int or []
    SKIP_ROWS = []
    # Specify index of rows for test set if TT. list of int or []
    TEST_SET_IND = list(randint.rvs(0, 918, size=184))

    # # Wine data - classification, 3 classes
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'wine'+'_'+TYPE
    # # Specify task KIND. string (clf, reg)
    # KIND = 'clf'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/wine_20221122.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id']
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'alcohol',
    #     'malic_acid',
    #     'ash',
    #     'alcalinity_of_ash',
    #     'magnesium',
    #     'total_phenols',
    #     'flavanoids',
    #     'nonflavanoid_phenols',
    #     'proanthocyanins',
    #     'color_intensity',
    #     'hue',
    #     'od280_od315_of_diluted_wines',
    #     'proline']
    # # Specify categorical predictor names. list of string or []
    # X_CAT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'target']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 178, size=36))

    ###########################################################################

    # Create results directory path -------------------------------------------
    path_to_results = 'res_'+KIND+'_'+ANALYSIS_NAME+'_'+ESTIMATOR_NAME

    # Create task variable ----------------------------------------------------
    task = {'TYPE': TYPE,
            'N_JOBS': N_JOBS,
            'N_REP_OUTER_CV': N_REP_OUTER_CV,
            'N_SAMPLES_INNER_CV': N_SAMPLES_INNER_CV,
            'TST_SIZE_FRAC': TST_SIZE_FRAC,
            'N_SAMPLES_RS': N_SAMPLES_RS,
            'ESTIMATOR_NAME': ESTIMATOR_NAME,
            'MAX_SAMPLES_SHAP': MAX_SAMPLES_SHAP,
            'MAX_SAMPLES_SHAP_BACKGROUND': MAX_SAMPLES_SHAP_BACKGROUND,
            'SHAP_INTERACTIONS': SHAP_INTERACTIONS,
            'DROP_NAN': DROP_NAN,
            'ANALYSIS_NAME': ANALYSIS_NAME,
            'KIND': KIND,
            'PATH_TO_DATA': PATH_TO_DATA,
            'G_NAME': G_NAME,
            'X_CON_NAMES': X_CON_NAMES,
            'X_CAT_NAMES': X_CAT_NAMES,
            'Y_NAMES': Y_NAMES,
            'SKIP_ROWS': SKIP_ROWS,
            'TEST_SET_IND': TEST_SET_IND,
            'path_to_results': path_to_results}

    # Create results directory ------------------------------------------------
    create_dir(path_to_results)

    # Copy this python script to results directory ----------------------------
    shutil.copy('ml_2_analysis.py', path_to_results+'/ml_2_analysis.py')

    # Load data ---------------------------------------------------------------
    # Load groups from excel file
    g = pd.read_excel(PATH_TO_DATA,
                      sheet_name=SHEET_NAME,
                      header=0,
                      usecols=G_NAME,
                      dtype=np.float64,
                      skiprows=SKIP_ROWS)
    # Load predictors from excel file
    x = pd.read_excel(PATH_TO_DATA,
                      sheet_name=SHEET_NAME,
                      header=0,
                      usecols=X_CON_NAMES+X_CAT_NAMES,
                      dtype=np.float64,
                      skiprows=SKIP_ROWS)
    # Load targets from excel file
    y = pd.read_excel(PATH_TO_DATA,
                      sheet_name=SHEET_NAME,
                      header=0,
                      usecols=Y_NAMES,
                      dtype=np.float64,
                      skiprows=SKIP_ROWS)

    # Drop rows with nans -----------------------------------------------------
    if task['DROP_NAN']:
        # Drop rows with nans
        g, x, y = drop_nan_rows(g, x, y)

    # Get one-hot-encoding categories and names but don't do actual encoding --
    # Instanciate one-hot-encoder
    ohe = OneHotEncoder(categories='auto',
                        drop='if_binary',
                        sparse_output=False,
                        dtype=np.float64,
                        handle_unknown='error',
                        min_frequency=None,
                        max_categories=None)
    # Fit one-hot-encoder
    ohe.fit(x[task['X_CAT_NAMES']])
    # Get one-hot-encoder categories
    task['ohe_categories'] = ohe.categories_
    # Get one-hot-encoder category names
    task['x_ohe_names'] = list(ohe.get_feature_names_out())
    # All predictor names
    task['x_names'] = task['X_CON_NAMES']+task['x_ohe_names']

    # Cross-validation --------------------------------------------------------
    # Iterate over prediction targets (y names)
    for i_y, y_name in enumerate(Y_NAMES):
        # Add prediction target index to task
        task['i_y'] = i_y
        # Add prediction target name to task
        task['y_name'] = [y_name]
        # Get current target
        yi = y[y_name].to_frame()
        # Cross-validation
        if TYPE == 'CV':
            # Run cross-validation
            cross_validation(task, g, x, yi)
        # Switch Type of analysis
        elif TYPE == 'TT':
            # Run train-test split
            train_test_split(task, g, x, yi)
        # Other
        else:
            # Raise error
            raise TypeError('Type not found.')


if __name__ == '__main__':
    main()
