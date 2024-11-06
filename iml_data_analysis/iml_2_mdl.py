# -*- coding: utf-8 -*-
'''
Interpretable Machine-Learning - Modelling (MDL)
v817
@author: Dr. David Steyrl david.steyrl@univie.ac.at
'''

import math as mth
import numpy as np
import os
import pandas as pd
import pickle as pkl
import shutil
import warnings
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from shap import Explanation
from shap.explainers import Tree as TreeExplainer
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import shuffle
from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from time import time


def create_dir(path: str) -> None:
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

    # Create dir of not existing ----------------------------------------------
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)

    # Return None -------------------------------------------------------------
    return


def prepare(task: dict) -> tuple:
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

    # Make preprocessing pipe -------------------------------------------------
    # Instatiate target-encoder
    te = TargetEncoder(
        categories=task['te_categories'],
        target_type='continuous',
        smooth='auto',
        cv=task['N_CV_FOLDS'],
        shuffle=True,
        random_state=None)
    # Get categorical predictors for target-encoder
    coltrans = ColumnTransformer(
        [('con_pred', 'passthrough', task['X_CON_NAMES']),
         ('bin_pred', 'passthrough', task['X_CAT_BIN_NAMES']),
         ('mult_pred', te, task['X_CAT_MULT_NAMES']),
         ],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Pipeline
    pre_pipe = Pipeline(
        [('coltrans', coltrans), ('std_scaler', StandardScaler())],
        memory=None,
        verbose=False)

    # Make estimator ----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Estimator
        estimator = LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective='huber',
            min_split_gain=0.0,
            min_child_weight=0.0,
            min_child_samples=10,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type='gain',
            **{'data_random_seed': None,
               'data_sample_strategy': 'bagging',
               'extra_seed': None,
               'feature_fraction_seed': None,
               'feature_pre_filter': False,
               'force_col_wise': True,
               'min_data_in_bin': 1,
               'use_quantized_grad': True,
               'verbosity': -1,
               })
        # Add scaler to the estimator
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            func=None,
            inverse_func=None,
            check_inverse=True)
        # Search space
        space = {
            'estimator__regressor__colsample_bytree': uniform(0.1, 0.9),
            'estimator__regressor__extra_trees': [True, False],
            'estimator__regressor__reg_lambda': loguniform(0.1, 100),
            }
    # Classification
    elif task['OBJECTIVE'] == 'classification':
        # Estimator
        estimator = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective='multiclass',
            class_weight='balanced',
            min_split_gain=0.0,
            min_child_weight=0.0,
            min_child_samples=10,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type='gain',
            **{'data_random_seed': None,
               'data_sample_strategy': 'bagging',
               'extra_seed': None,
               'feature_fraction_seed': None,
               'feature_pre_filter': False,
               'force_col_wise': True,
               'min_data_in_bin': 1,
               'num_class': task['n_classes'],
               'use_quantized_grad': True,
               'verbosity': -1,
               })
        # Search space
        space = {
            'estimator__colsample_bytree': uniform(0.1, 0.9),
            'estimator__extra_trees': [True, False],
            'estimator__reg_lambda': loguniform(0.1, 100),
            }
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Make full pipeline ------------------------------------------------------
    # Analyis pipeline
    pipe = Pipeline(
        [('preprocessing', pre_pipe),
         ('estimator', estimator)],
        memory=None,
        verbose=False).set_output(transform='pandas')

    # Return pipe and space ---------------------------------------------------
    return pipe, space


def split_data(df: pd.DataFrame, i_trn: np.ndarray,
               i_tst: np.ndarray) -> tuple:
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

    # Split dataframe via index -----------------------------------------------
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

    # Return train test dataframes --------------------------------------------
    return df_trn, df_tst


def print_tune_summary(task: dict, i_cv: int, hp_params: dict,
                       hp_score: dict) -> None:
    '''
    Print best paramters and related score to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current cv repetition.
    hp_params : dictionary
        Best hyper params found.
    hp_score : dictionary
        Score for best hyper params found.

    Returns
    -------
    None.
    '''

    # Print analysis name
    print('Analysis: '+task['ANALYSIS_NAME'])
    # Print data set
    print('Dataset: '+task['PATH_TO_DATA'])
    # Print prediction target
    print('Predicting: '+task['y_name'])
    # Cross-validation --------------------------------------------------------
    if task['TYPE'] == 'CV':
        # Regression
        if task['OBJECTIVE'] == 'regression':
            # Print general information
            print(
                str(task['i_y'])+'.'+str(i_cv)+' | ' +
                'n rep outer cv: '+str(task['n_rep_outer_cv'])+' | ' +
                'n rep inner cv: '+str(task['n_rep_inner_cv'])+' | ' +
                'best R²: '+str(np.round(hp_score, decimals=4)))
        # Classification
        elif task['OBJECTIVE'] == 'classification':
            # Print general information
            print(
                str(task['i_y'])+'.'+str(i_cv)+' | ' +
                'n rep outer cv: '+str(task['n_rep_outer_cv'])+' | ' +
                'n rep inner cv: '+str(task['n_rep_inner_cv'])+' | ' +
                'best acc: '+str(np.round(hp_score, decimals=4)))
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')
    # Train-Test split --------------------------------------------------------
    elif task['TYPE'] == 'TT':
        # Regression
        if task['OBJECTIVE'] == 'regression':
            # Print general information
            print(
                str(task['i_y'])+'.'+str(i_cv)+' | ' +
                'n rep inner cv: '+str(task['n_rep_inner_cv'])+' | ' +
                'best R²: '+str(np.round(hp_score, decimals=4)))
        # Classification
        elif task['OBJECTIVE'] == 'classification':
            # Print general information
            print(
                str(task['i_y'])+'.'+str(i_cv)+' | ' +
                'n rep inner cv: '+str(task['n_rep_inner_cv'])+' | ' +
                'best acc: '+str(np.round(hp_score, decimals=4)))
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')
    # Other -------------------------------------------------------------------
    else:
        # Raise error
        raise ValueError('TYPE not found.')
    # Print best hyperparameter and related score for regression task
    print(str(hp_params))

    # Return None -------------------------------------------------------------
    return


def tune_pipe(task: dict, i_cv: int, pipe: Pipeline, space: dict,
              g_trn: np.ndarray, x_trn: np.ndarray,
              y_trn: np.ndarray) -> tuple:
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

    # Get scorer --------------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # R² score
        scorer = 'r2'
    # Classification
    elif task['OBJECTIVE'] == 'classification':
        # Balanced accuracy for classification
        scorer = 'balanced_accuracy'
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Tune analysis pipeline --------------------------------------------------
    # Choose n_repeats to approx N_SAMPLES_INNER_CV predictions
    task['n_rep_inner_cv'] = mth.ceil(task['N_PRED_INNER_CV'] / g_trn.shape[0])
    # Instatiate random parameter search
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=task['N_SAMPLES_RS'],
        scoring=scorer,
        n_jobs=task['N_JOBS'],
        refit=True,
        cv=RepeatedGroupKFold(n_splits=task['N_CV_FOLDS'],
                              n_repeats=task['n_rep_inner_cv'],
                              random_state=None),
        verbose=0,
        pre_dispatch='2*n_jobs',
        random_state=None,
        error_score=0,
        return_train_score=False)
    # Random search for best parameter
    search.fit(x_trn, y_trn.squeeze(), groups=g_trn)
    # Print tune summary
    print_tune_summary(
        task,
        i_cv,
        search.best_params_,
        search.best_score_)

    # Return tuned analysis pipe ----------------------------------------------
    return search.best_estimator_, search.best_params_


def score_predictions(task: dict, pipe: Pipeline, x_tst: np.ndarray,
                      y_tst: np.ndarray, i_tst: np.ndarray,
                      y: np.ndarray) -> dict:
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
    i_tst : ndarray (n_samples)
        Index of testing samples.
    y : ndarray
        All available target data to compute true class weights for scoring.

    Returns
    -------
    scores : dict
        Returns scoring results. MAE, MSE and R² if task is regression.
        ACC and true class weights if task is classification.
    '''

    # Predict -----------------------------------------------------------------
    # Predict test samples
    y_pred = pipe.predict(x_tst)

    # Score results -----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of mse
        mse = mean_squared_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {
            'y_true': y_tst.squeeze().to_numpy(),
            'y_pred': y_pred,
            'y_ind': i_tst,
            'mae': mae,
            'mse': mse,
            'r2': r2}
    # Classification
    elif task['OBJECTIVE'] == 'classification':
        # Calculate model fit in terms of acc
        acc = balanced_accuracy_score(y_tst, y_pred)
        # Results
        scores = {
            'y_true': y_tst.squeeze().to_numpy(),
            'y_pred': y_pred,
            'y_ind': i_tst,
            'acc': acc}
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Return scores -----------------------------------------------------------
    return scores


def get_explainations(task: dict, pipe: Pipeline, x_trn: np.ndarray,
                      x_tst: np.ndarray) -> Explanation:
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

    # Get SHAP test data ------------------------------------------------------
    # If task is CV
    if task['TYPE'] == 'CV':
        # Get max samples shap
        task['max_samples_shap'] = min(
            x_tst.shape[0],
            mth.ceil(task['N_SAMPLES_SHAP']/(task['n_rep_outer_cv'] *
                                             task['N_CV_FOLDS']))
            )
    # If task is TT
    if task['TYPE'] == 'TT':
        # Get max samples shap
        task['max_samples_shap'] = min(x_tst.shape[0], task['N_SAMPLES_SHAP'])
    # Subsample test data
    x_tst_shap_orig = x_tst.sample(
        n=task['max_samples_shap'],
        random_state=314,
        ignore_index=True)
    # Transform shap test data
    x_tst_shap = pipe[0].transform(x_tst_shap_orig)

    # Explainer and Explainations ---------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Get predictor
        predictor = pipe[1].regressor_
    # Classification
    elif task['OBJECTIVE'] == 'classification':
        # Get predictor
        predictor = pipe[1]
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')
    # Get explainer
    explainer = TreeExplainer(
        predictor,
        data=None,
        model_output='raw',
        feature_perturbation='tree_path_dependent',
        feature_names=None,
        approximate=False)
    # Get explainations with interactions
    if task['SHAP_WITH_INTERACTIONS']:
        # Get shap values
        shap_explainations = explainer(
            x_tst_shap,
            interactions=True,
            check_additivity=False)
    # Get explainations without interactions
    elif not task['SHAP_WITH_INTERACTIONS']:
        # Get shap values
        shap_explainations = explainer(
            x_tst_shap,
            interactions=False,
            check_additivity=False)
    # Other
    else:
        # Raise error
        raise ValueError('Invalid value for SHAP_WITH_INTERACTIONS.')

    # Post process shap_explainations -----------------------------------------
    if task['SHAP_USE_TARGET_ENC_VALUES']:
        # Initialize x_test for shap with target encoding
        x_tst_shap_enc = x_tst_shap_orig
        # Transform the original x_tst for shap with the target encoder
        x_tst_shap_mult = pipe[0][0].transform(x_tst_shap_orig)
        # Loop over the cat mult predictors
        for x_cat_mult_name in task['X_CAT_MULT_NAMES']:
            # Set in the encoded values in the x_tst for shap with target enc
            x_tst_shap_enc[x_cat_mult_name] = x_tst_shap_mult[x_cat_mult_name]
        # Replace scaled data in shap explainations with target enc
        shap_explainations.data = x_tst_shap_enc
    else:
        # Replace scaled data in shap explainations with original
        shap_explainations.data = x_tst_shap_orig
    # If regression
    if task['OBJECTIVE'] == 'regression':
        # Rescale shap values from scaled data to original space
        shap_explainations.values = (
            shap_explainations.values*pipe[1].transformer_.scale_[0])
        # Rescale shap base values from scaled data to original space
        shap_explainations.base_values = (
            (shap_explainations.base_values*pipe[1].transformer_.scale_[0]) +
            pipe[1].transformer_.mean_[0])

    # Return shap explainations -----------------------------------------------
    return shap_explainations


def s2p(path_save: str, variable: str) -> None:
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

    # Save --------------------------------------------------------------------
    # Save variable as pickle file
    with open(path_save, 'wb') as filehandle:
        # store the data as binary data stream
        pkl.dump(variable, filehandle)

    # Return None -------------------------------------------------------------
    return


def print_current_results(task: dict, t_start: time, scores: dict,
                          scores_sh: dict) -> None:
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

    # Print results -----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Print current R2
        print(
            'Current CV loop R²: '+str(np.round(
                scores[-1]['r2'], decimals=4)))
        # Print running mean R2
        print(
            'Running mean R²: '+str(np.round(
                np.mean([i['r2'] for i in scores]), decimals=4)))
        # Print running mean shuffle R2
        print(
            'Running shuffle mean R²: '+str(np.round(
                np.mean([i['r2'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print(
            'Elapsed time: '+str(np.round(
                time() - t_start, decimals=1)), end='\n\n')
    # Classification
    elif task['OBJECTIVE'] == 'classification':
        # Print current acc
        print(
            'Current CV loop acc: '+str(np.round(
                scores[-1]['acc'], decimals=4)))
        # Print running mean acc
        print(
            'Running mean acc: '+str(np.round(
                np.mean([i['acc'] for i in scores]), decimals=4)))
        # Print running mean shuffle acc
        print(
            'Running shuffle mean acc: '+str(np.round(
                np.mean([i['acc'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print(
            'Elapsed time: '+str(np.round(
                time() - t_start, decimals=1)), end='\n\n')
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Return None -------------------------------------------------------------
    return


def cross_validation(task: dict, g: pd.DataFrame, x: pd.DataFrame,
                     y: pd.DataFrame) -> None:
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
    # Initialize best pipes list
    best_pipes = []
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
    # Calculate number of repetition for outer CV
    task['n_rep_outer_cv'] = mth.ceil(task['N_PRED_OUTER_CV']/g.shape[0])
    # Instatiate main cv splitter with fixed random state for comparison
    cv = RepeatedGroupKFold(
        n_splits=task['N_CV_FOLDS'],
        n_repeats=task['n_rep_outer_cv'],
        random_state=314)
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
        # Store best pipe
        best_pipes.append(pipe)

        # Analyze -------------------------------------------------------------
        # Score predictions
        scores.append(score_predictions(task, pipe, x_tst, y_tst, i_tst, y))
        # SHAP explainations
        explainations.append(get_explainations(task, pipe, x_trn, x_tst))

        # Shuffle data analyze ------------------------------------------------
        # Clone pipe
        pipe_sh = clone(pipe)
        # Refit pipe with shuffled targets
        pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
        # Score predictions
        scores_sh.append(
            score_predictions(task, pipe_sh, x_tst, y_tst, i_tst, y))
        # SHAP explainations
        explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

        # Compile and save intermediate results and task ----------------------
        # Create results
        results = {
            'best_params': best_params,
            'best_pipes': best_pipes,
            'scores': scores,
            'explainations': explainations,
            'scores_sh': scores_sh,
            'explainations_sh': explainations_sh
            }
        # Save results as pickle file
        s2p(task['save_path']+'_results.pickle', results)
        # Save task as pickle file
        s2p(task['save_path']+'_task.pickle', task)

        # Print current results -----------------------------------------------
        print_current_results(task, t_start, scores, scores_sh)

    # Return results ----------------------------------------------------------
    return results


def train_test_split(task: dict, g: pd.DataFrame, x: pd.DataFrame,
                     y: pd.DataFrame) -> None:
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
    # Initialize best pipes list
    best_pipes = []
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
    # Store best pipe
    best_pipes.append(pipe)

    # Analyze -----------------------------------------------------------------
    # Score predictions
    scores.append(score_predictions(task, pipe, x_tst, y_tst, i_tst, y))
    # SHAP explainations
    explainations.append(get_explainations(task, pipe, x_trn, x_tst))

    # Shuffle data analyze ----------------------------------------------------
    # Clone pipe
    pipe_sh = clone(pipe)
    # Refit pipe with shuffled targets
    pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
    # Score predictions
    scores_sh.append(score_predictions(task, pipe_sh, x_tst, y_tst, i_tst, y))
    # SHAP explainations
    explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

    # Compile and save intermediate results and task --------------------------
    # Create results
    results = {
        'best_params': best_params,
        'best_pipes': best_pipes,
        'scores': scores,
        'explainations': explainations,
        'scores_sh': scores_sh,
        'explainations_sh': explainations_sh
        }
    # Save results as pickle file
    s2p(task['save_path']+'_results.pickle', results)
    # Save task as pickle file
    s2p(task['save_path']+'_task.pickle', task)

    # Print current results ---------------------------------------------------
    print_current_results(task, t_start, scores, scores_sh)

    # Return results ----------------------------------------------------------
    return results


def main() -> None:
    '''
    Main function of Interpretable Machine-Learning.

    Returns
    -------
    None.
    '''

    ###########################################################################
    # Specify analysis tasks
    ###########################################################################

    # 1. Specify task ---------------------------------------------------------

    # Type of analysis. str (default: CV)
    # Repeated Cross-validation: CV
    # Single Train-Test split: TT
    TYPE = 'CV'
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    N_JOBS = -2
    # Number of folds in CV. int (default: 5)
    N_CV_FOLDS = 5
    # Number of predictions in outer CV (if TYPE='CV'). int (default: 10000)
    N_PRED_OUTER_CV = 1000
    # Number of tries in random search. int (default: 100)
    N_SAMPLES_RS = 100
    # Number of predictions in inner CV. int (default: 1000)
    N_PRED_INNER_CV = 1000
    # Number of samples SHAP. int (default: 10000).
    N_SAMPLES_SHAP = 10000
    # Get SHAP interactions. bool (default: True)
    SHAP_WITH_INTERACTIONS = True
    # Use target encoded or original values in SHAP. bool (default: True)
    SHAP_USE_TARGET_ENC_VALUES = True

    # 2. Specify data ---------------------------------------------------------

    # # Cancer data - classification 2 class
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'cancer'
    # # Specify path to data. string
    # PATH_TO_DATA = 'sample_data/cancer_20240806.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = 'classification'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id',
    #     ]
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
    #     'worst_fractal_dimension',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'benign_tumor',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 569, size=114, random_state=314))

    # # Diabetes data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'diabetes'
    # # Specify path to data. string
    # PATH_TO_DATA = 'sample_data/diabetes_20240806.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = 'regression'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id',
    #     ]
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'age',
    #     'bmi',
    #     'bp',
    #     's1_tc',
    #     's2_ldl',
    #     's3_hdl',
    #     's4_tch',
    #     's5_ltg',
    #     's6_glu',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = [
    #     'gender',
    #     ]
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'progression',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 442, size=88, random_state=314))

    # Employee data - classification 2 class
    # Specifiy an analysis name
    ANALYSIS_NAME = 'employee'
    # Specify path to data. string
    PATH_TO_DATA = 'sample_data/employee_20240806.xlsx'
    # Specify sheet name. string
    SHEET_NAME = 'data'
    # Specify task OBJECTIVE. string (classification, regression)
    OBJECTIVE = 'classification'
    # Specify grouping for CV split. list of string
    G_NAME = [
        'sample_id',
        ]
    # Specify continous predictor names. list of string or []
    X_CON_NAMES = [
        'age',
        'distance_from_home',
        'environment_satisfaction',
        'job_satisfaction',
        'monthly_income',
        'num_companies_worked',
        'stock_option_level',
        'training_times_last_year',
        'total_working_years',
        'work_life_balance',
        'years_at_company',
        'years_since_last_promotion',
        'years_with_curr_manager',
        ]
    # Specify binary categorical predictor names. list of string or []
    X_CAT_BIN_NAMES = [
        'gender',
        'over_time',
        ]
    # Specify multi categorical predictor names. list of string or []
    X_CAT_MULT_NAMES = [
        'marital_status',
        ]
    # Specify target name(s). list of strings or []
    Y_NAMES = [
        'attrition',
        ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []
    # Specify index of rows for test set if TT. list of int or []
    TEST_SET_IND = list(randint.rvs(0, 1470, size=294, random_state=314))

    # # Housing data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'housing'
    # # Specify path to data. string
    # PATH_TO_DATA = 'sample_data/housing_20240806.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = 'regression'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id',
    #     ]
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'median_income',
    #     'house_age',
    #     'average_rooms',
    #     'average_bedrooms',
    #     'population',
    #     'average_occupation',
    #     'latitude',
    #     'longitude',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = [
    #     'ocean_proximity',
    #     ]
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'median_house_value',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 20640, size=4128, random_state=314))

    # # Radon data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'radon'
    # # Specify path to data. string
    # PATH_TO_DATA = 'sample_data/radon_20240806.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = 'regression'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id',
    #     ]
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'uppm',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = [
    #     'basement',
    #     'floor',
    #     ]
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = [
    #     'county_code',
    #     'region',
    #     'room',
    #     'zip',
    #     ]
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'log_radon',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 878, size=176, random_state=314))

    # # Wine data - classification 3 class
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'wine'
    # # Specify path to data. string
    # PATH_TO_DATA = 'sample_data/wine_20240806.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = 'classification'
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     'sample_id',
    #     ]
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
    #     'proline',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'maker',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 178, size=36, random_state=314))

    ###########################################################################

    # Add to analysis name ----------------------------------------------------
    # If shap with interactions
    if SHAP_WITH_INTERACTIONS:
        # Update string
        ANALYSIS_NAME = ANALYSIS_NAME+'_'+TYPE+'_'+'inter'
    # If shap without interactions
    elif not SHAP_WITH_INTERACTIONS:
        # Update string
        ANALYSIS_NAME = ANALYSIS_NAME+'_'+TYPE
    # Other
    else:
        # Raise error
        raise ValueError('SHAP_WITH_INTERACTIONS can be True or False only.')

    # Create results directory path -------------------------------------------
    path_to_results = 'res_iml_'+ANALYSIS_NAME

    # Create results directory ------------------------------------------------
    create_dir(path_to_results)

    # Create task variable ----------------------------------------------------
    task = {
        'TYPE': TYPE,
        'N_JOBS': N_JOBS,
        'N_CV_FOLDS': N_CV_FOLDS,
        'N_PRED_OUTER_CV': N_PRED_OUTER_CV,
        'N_PRED_INNER_CV': N_PRED_INNER_CV,
        'N_SAMPLES_RS': N_SAMPLES_RS,
        'N_SAMPLES_SHAP': N_SAMPLES_SHAP,
        'SHAP_WITH_INTERACTIONS': SHAP_WITH_INTERACTIONS,
        'SHAP_USE_TARGET_ENC_VALUES': SHAP_USE_TARGET_ENC_VALUES,
        'ANALYSIS_NAME': ANALYSIS_NAME,
        'PATH_TO_DATA': PATH_TO_DATA,
        'SHEET_NAME': SHEET_NAME,
        'OBJECTIVE': OBJECTIVE,
        'G_NAME': G_NAME,
        'X_CON_NAMES': X_CON_NAMES,
        'X_CAT_BIN_NAMES': X_CAT_BIN_NAMES,
        'X_CAT_MULT_NAMES': X_CAT_MULT_NAMES,
        'Y_NAMES': Y_NAMES,
        'SKIP_ROWS': SKIP_ROWS,
        'TEST_SET_IND': TEST_SET_IND,
        'path_to_results': path_to_results,
        'x_names': X_CON_NAMES+X_CAT_BIN_NAMES+X_CAT_MULT_NAMES,
        }

    # Copy this python script to results directory ----------------------------
    shutil.copy('iml_2_mdl.py', path_to_results+'/iml_2_mdl.py')

    # Load data ---------------------------------------------------------------
    # Load groups from excel file
    G = pd.read_excel(
        task['PATH_TO_DATA'],
        sheet_name=task['SHEET_NAME'],
        header=0,
        usecols=task['G_NAME'],
        dtype=np.float64,
        skiprows=task['SKIP_ROWS'])
    # Load predictors from excel file
    X = pd.read_excel(
        task['PATH_TO_DATA'],
        sheet_name=task['SHEET_NAME'],
        header=0,
        usecols=task['x_names'],
        dtype=np.float64,
        skiprows=task['SKIP_ROWS'])
    # Reindex x to x_names
    X = X.reindex(task['x_names'], axis=1)
    # Load targets from excel file
    Y = pd.read_excel(
        task['PATH_TO_DATA'],
        sheet_name=task['SHEET_NAME'],
        header=0,
        usecols=task['Y_NAMES'],
        dtype=np.float64,
        skiprows=task['SKIP_ROWS'])

    # Modelling and testing ---------------------------------------------------
    # Iterate over prediction targets (Y_NAMES)
    for i_y, y_name in enumerate(Y_NAMES):
        # Add prediction target index to task
        task['i_y'] = i_y
        # Add prediction target name to task
        task['y_name'] = y_name
        # Make save path
        save_path = task['path_to_results']+'/'+task['y_name']
        # Add save path to task
        task['save_path'] = save_path

        # Deal with NaNs in the target ----------------------------------------
        # Get current target and remove NaNs
        y = Y[y_name].to_frame().dropna()
        # Use y index for groups and reset index
        g = G.reindex(index=y.index).reset_index(drop=True)
        # Use y index for predictors and reset index
        x = X.reindex(index=y.index).reset_index(drop=True)
        # Reset index of target
        y = y.reset_index(drop=True)
        # Raise Warning if samples were dropped because of NaNs in target
        if y.shape[0] < Y.shape[0]:
            # Warning
            warnings.warn(
                'Warning: ' +
                str(Y.shape[0]-y.shape[0]) +
                ' samples were dropped due to NaNs in ' +
                y_name+'.', UserWarning)

        # Save fianl data set -------------------------------------------------
        # Save group data
        g.to_excel(task['save_path']+'_g.xlsx', index=False)
        # Save predictor data
        x.to_excel(task['save_path']+'_x.xlsx', index=False)
        # Save target data
        y.to_excel(task['save_path']+'_y.xlsx', index=False)

        # Get target-encoding categories but don't do encoding ----------------
        # If multi categorical predictors
        if task['X_CAT_MULT_NAMES']:
            # Instatiate target-encoder
            te = TargetEncoder(
                categories='auto',
                target_type='continuous',
                smooth='auto',
                cv=task['N_CV_FOLDS'],
                shuffle=True,
                random_state=None)
            # Fit target-encoder
            te.fit(x[task['X_CAT_MULT_NAMES']], y.squeeze())
            # Get target-encoder categories
            task['te_categories'] = te.categories_
        # Other
        else:
            # Set target-encoder categories to empty
            task['te_categories'] = []

        # Get number of classes if task is classification ---------------------
        # Classification
        if task['OBJECTIVE'] == 'classification':
            # Number of unique classes in prediction target
            task['n_classes'] = y.nunique()[task['y_name']]
        # Regression
        elif task['OBJECTIVE'] == 'regression':
            # Set number of classes to -1 for compatibility
            task['n_classes'] = -1

        # Run analysis --------------------------------------------------------
        # Cross-validation
        if TYPE == 'CV':
            # Run cross-validation
            results = cross_validation(task, g, x, y)
        # Switch Type of analysis
        elif TYPE == 'TT':
            # Run train-test split
            results = train_test_split(task, g, x, y)
        # Other
        else:
            # Raise error
            raise ValueError('Analysis type not found.')

    # Return None -------------------------------------------------------------
    return results


if __name__ == '__main__':
    main()
