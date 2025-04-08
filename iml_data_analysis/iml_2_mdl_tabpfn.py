# -*- coding: utf-8 -*-
"""
Interpretable Machine-Learning - Modelling (MDL-tabpfn)
v001
@author: david.steyrl@univie.ac.at
"""

import logging
import math as mth
import numba
import numpy as np
import os
import pandas as pd
import pickle as pkl
import shutil
import subprocess
import warnings
from scipy.stats import randint
from shap import Explanation
from shap.links import identity
from shap.maskers import Partition
from shap import PermutationExplainer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import shuffle
from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
from time import sleep
from time import time
from typing import Union

# Supress warnings about classes not in y_true
os.environ["PYTHONWARNINGS"] = "ignore:y_pred contains classes not in y_true:::"
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def get_pip_requirements() -> str:
    """
    Retrieve the current pip requirements as a string.

    Parameters
    ----------
    None

    Returns
    -------
    str: The YAML-formatted configuration of the current pip requirements.

    Raises
    ------
    RuntimeError: If the subprocess fails to run the `pip freeze` command.
    Exception: If unexpected error occurred.
    """
    try:
        # Run the 'pip freeze' command
        pip_requirements = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as a string
            shell=True,
        )
        # If command was successful
        if pip_requirements.returncode == 0:
            # Return the pip requirements string
            return pip_requirements.stdout
        # If command was not successful
        else:
            # Raise error
            raise RuntimeError(f"Failed to run 'pip freeze': {pip_requirements.stderr}")
    except Exception as e:
        # Raise exception
        raise e


def get_estimator(task: dict) -> Union[TabPFNRegressor, TabPFNClassifier]:
    """
    Prepare analysis pipeline and search space.

    Parameters
    ----------
    task: dict
        Dictionary containing task details.

    Returns
    -------
    Union[TabPFNRegressor, TabPFNClassifier]: A tuple containing the prepared pipeline
        and search space.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Instatiate target-encoder
        target_encoder = TargetEncoder(
            categories="auto",
            target_type="continuous",
            smooth="auto",
            cv=task["N_CV_FOLDS"],
            shuffle=True,
            random_state=None,
        ).set_output(transform="pandas")
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Instatiate target-encoder
        target_encoder = TargetEncoder(
            categories="auto",
            target_type="binary" if task["n_classes"] == 2 else "multiclass",
            smooth="auto",
            cv=task["N_CV_FOLDS"],
            shuffle=True,
            random_state=None,
        ).set_output(transform="pandas")
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Names of columns for target encoding
    names_target_encoder = [task["X_NAMES"][i] for i in task["TARGET_ENCODING_IND"]]
    # Names of columns to passthrough
    names_passthrough = [i for i in task["X_NAMES"] if i not in names_target_encoder]
    # Use target-encoder for multi categorical predictors
    column_transformer = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", names_passthrough),
            ("target_encoder", target_encoder, names_target_encoder),
        ],
        remainder="drop",
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Get predictor
        predictor = TabPFNRegressor(
            n_estimators=1,
            categorical_features_indices=None,
            softmax_temperature=0.9,
            average_before_softmax=False,
            device=task["DEVICE"],
            fit_mode="fit_preprocessors",
            random_state=None,
            n_jobs=-2,
        )
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Get predictor
        predictor = TabPFNClassifier(
            n_estimators=1,
            categorical_features_indices=None,
            softmax_temperature=0.9,
            balance_probabilities=True,
            average_before_softmax=False,
            device=task["DEVICE"],
            fit_mode="fit_preprocessors",
            random_state=None,
            n_jobs=-2,
        )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Analyis pipeline
    estimator = Pipeline(
        [("column_transformer", column_transformer), ("predictor", predictor)],
        memory=None,
        verbose=False,
    ).set_output(transform="pandas")
    # Return estimator
    return estimator


def split_data(df: pd.DataFrame, i_trn: np.ndarray, i_tst: np.ndarray) -> tuple:
    """
    Split a DataFrame into training and testing datasets.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data to split.
    i_trn: np.ndarray
        Array of indices for the training set.
    i_tst: np.ndarray
        Array of indices for the testing set.

    Returns
    -------
    tuple: Tuple containing the training DataFrame (df_trn) and testing DataFrame
        (df_tst).

    Raises
    ------
    None
    """
    # If empty input
    if df.empty:
        # Return empty training and testing DataFrames
        return pd.DataFrame(), pd.DataFrame()
    # Perform the split for training
    df_trn = df.loc[df.index.intersection(set(i_trn))]
    # Perform the split for testing
    df_tst = df.loc[df.index.intersection(set(i_tst))]
    # Return train and test dataframes
    return df_trn, df_tst


def score_predictions(
    task: dict,
    estimator: Union[TabPFNRegressor, TabPFNClassifier],
    x_tst: np.ndarray,
    y_tst: np.ndarray,
    i_tst: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Evaluate the predictions of a pipeline on a test dataset based on the specified
    task.

    Parameters
    ----------
    task: dict
        Dictionary containing task-specific configuration and metadata, such as the
        objective (e.g., "regression" or "classification") and scoring criteria.
    estimator: Union[TabPFNRegressor, TabPFNClassifier]
        A TabPFNRegressor or TabPFNClassifier used to make predictions.
    x_tst: np.ndarray
        Feature matrix (n_samples, n_features) for the test dataset.
    y_tst: np.ndarray
        Target values (n_samples) for the test dataset.
    i_tst: np.ndarray
        Indices of the samples in the test set.
    y: np.ndarray
        Complete target values (n_samples) for all available data, used to compute true
        class weights for classification tasks.

    Returns
    -------
    dict: A dictionary containing evaluation metrics:
        - For regression tasks: Mean Absolute Error (MAE), Mean Squared Error (MSE),
        and R² (coefficient of determination).
        - For classification tasks: Balanced accuracy (ACC) and true class weights.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # If regression
    if task["OBJECTIVE"] == "regression":
        # Predict test samples
        y_pred = estimator.predict(x_tst)
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "y_ind": i_tst,
            "mae": mae,
            "r2": r2,
        }
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Predict test samples
        y_pred = estimator.predict(x_tst)
        # Predict test samples
        y_pred_proba = estimator.predict_proba(x_tst)
        # Calculate model fit in terms of acc
        acc = balanced_accuracy_score(y_tst, y_pred)
        # Results
        scores = {
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "y_ind": i_tst,
            "acc": acc,
        }
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Return scores
    return scores


@numba.njit
def safe_logit(x, eps=1e-10):
    """
    A Numba-jitted logit link function, useful for going from probability units to
    log-odds units.
    Handles both scalar and array inputs correctly in Numba nopython mode
    by using np.maximum/np.minimum instead of np.clip.

    Parameters
    ----------
    x: np.ndarray or float
        Input probability or array of probabilities.
    eps: float, optional
        Small epsilon value to prevent log(0) or division by zero. Defaults to 1e-10.

    Returns
    -------
    np.ndarray or float: The log-odds corresponding to x, with values clipped.
        Type matches the input type (scalar or array).

    Raises
    ------
    None
    """
    # Define bounds
    lower_bound = eps
    upper_bound = 1.0 - eps  # Use 1.0 for floating point
    # Clip using np.maximum and np.minimum. Numba handles these element-wise
    # functions correctly for both scalar and array inputs in nopython mode.
    clipped_x = np.maximum(lower_bound, np.minimum(x, upper_bound))
    # Logit transformation. np.log works element-wise for arrays
    # and also works for scalars within numba.
    # Using 1.0 ensures floating-point arithmetic.
    return np.log(clipped_x / (1.0 - clipped_x))


def get_explanations(
    task: dict,
    estimator: Union[TabPFNRegressor, TabPFNClassifier],
    x_trn: np.ndarray,
    x_tst: np.ndarray,
) -> Explanation:
    """
    Generate SHAP (SHapley Additive exPlanations) model explanations for feature
    importance. This function computes SHAP values to provide insights into model
    predictions by quantifying the contribution of each feature.

    References
    ----------
    - Molnar, Christoph. "Interpretable Machine Learning: A Guide for Making Black Box
      Models Explainable," 2019. https://christophm.github.io/interpretable-ml-book/.
    - Lundberg, Scott M., & Su-In Lee. "A unified approach to interpreting model
      predictions." Advances in Neural Information Processing Systems, 2017.
    - Lundberg, Scott M., Gabriel G. Erion, & Su-In Lee. "Consistent individualized
      feature attribution for tree ensembles." arXiv preprint arXiv:1802.03888 (2018).
    - Sundararajan, Mukund, & Amir Najmi. "The many Shapley values for model
      explanation." arXiv preprint arXiv:1908.08474 (2019).
    - Janzing, Dominik, Lenon Minorics, & Patrick Blöbaum. "Feature relevance
      quantification in explainable AI: A causality problem." arXiv preprint
      arXiv:1910.13413 (2019).
    - Slack, Dylan, et al. "Fooling LIME and SHAP: Adversarial attacks on post hoc
      explanation methods." Proceedings of the AAAI/ACM Conference on AI, Ethics, and
      Society, 2020.

    Parameters
    ----------
    task: dict
        Dictionary containing task-specific details, including model objectives and
        feature configurations.
    estimator: Union[TabPFNRegressor, TabPFNClassifier]
        A TabPFNRegressor or TabPFNClassifier used to make predictions.
    x_trn: np.ndarray
        Background dataset (n_samples, n_features) used by SHAP to establish a baseline
        for feature contribution analysis.
    x_tst: np.ndarray
        Test dataset (n_samples, n_features) on which SHAP values are computed to
        explain the model's predictions.

    Returns
    -------
    imp: Explanation
        SHAP explainer object containing feature importance values, which can be used
        to interpret the model's predictions and identify key contributing features.

    Raises
    ------
    ValueError: If TYPE is not CV and TT.
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Get SHAP test data ---
    # If CV
    if task["TYPE"] == "CV":
        # Get max samples shap
        task["max_samples_shap"] = min(
            x_tst.shape[0],
            mth.ceil(task["N_SAMPLES_SHAP"] / (task["n_rep_cv"] * task["N_CV_FOLDS"])),
        )
    # If TT
    elif task["TYPE"] == "TT":
        # Get max samples shap
        task["max_samples_shap"] = min(x_tst.shape[0], task["N_SAMPLES_SHAP"])
    else:
        # Raise error
        raise ValueError(f"TYPE is {task['TYPE']}.")
    # Subsample test data
    x_tst_shap = x_tst.sample(
        n=task["max_samples_shap"], random_state=1000, ignore_index=True
    )

    # --- Explainer and Explanations ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Get explainer
        explainer = PermutationExplainer(
            model=estimator.predict,
            masker=Partition(x_trn, max_samples=1000, clustering="correlation"),
            link=identity,
            feature_names=None,
            linearize_link=True,
            seed=None,
        )
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Get explainer
        explainer = PermutationExplainer(
            model=estimator.predict_proba,
            masker=Partition(x_trn, max_samples=1000, clustering="correlation"),
            link=safe_logit,
            feature_names=None,
            linearize_link=True,
            seed=None,
        )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Get shap explanations
    explanations = explainer(
        x_tst_shap,
        max_evals=2 * len(task["X_NAMES"]) + 1,
        silent=True,
    )

    # --- Return shap explanations ---
    return explanations


def log_current_results(
    task: dict, t_start: time, scores: dict, scores_sh: dict, i_cv: int = 0
) -> None:
    """
    Log current results to console.

    Parameters
    ----------
    task: dictionary
        Dictionary holding the task describtion variables.
    t_start : time
        Start time of the current cross-validation loop.
    scores: dict
        Scores dict.
    scores_sh: dict
        Scores with shuffled data dict.
    i_cv: int
        Currenter number of cv repetition.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Log analysis name
    logging.info(f"Analysis: {task['ANALYSIS_NAME']}")
    # Log data set
    logging.info(f"Dataset: {task['PATH_TO_DATA']}")
    # Log prediction target
    logging.info(f"Predicting: {task['y_name']}")
    # Log current number of repetition
    logging.info(f"{task['i_y']}.{i_cv} | n rep cv: {task['n_rep_cv']}")
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Log current R2
        logging.info(
            f"Current CV loop R²: {np.round(scores[-1]['r2'], decimals=4)}"
        )  # noqa
        # Log running mean R2
        logging.info(
            f"Running mean R²: {np.round(np.mean([i['r2'] for i in scores]), decimals=4)}"  # noqa
        )
        # Log running mean shuffle R2
        logging.info(
            f"Running shuffle mean R²: {np.round(np.mean([i['r2'] for i in scores_sh]), decimals=4)}"  # noqa
        )
        # Log elapsed time
        logging.info(f"Elapsed time: {np.round(time() - t_start, decimals=1)}\n")
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Log current acc
        logging.info(f"Current CV loop acc: {np.round(scores[-1]['acc'], decimals=4)}")
        # Log running mean acc
        logging.info(
            f"Running mean acc: {np.round(np.mean([i['acc'] for i in scores]), decimals=4)}"  # noqa
        )
        # Log running mean shuffle acc
        logging.info(
            f"Running shuffle mean acc: {np.round(np.mean([i['acc'] for i in scores_sh]), decimals=4)}"  # noqa
        )
        # Log elapsed time
        logging.info(f"Elapsed time: {np.round(time() - t_start, decimals=1)}\n")
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")


def single_train_test_split_predictions(
    task: dict,
    g: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.DataFrame,
    i_trn: list,
    i_tst: list,
    i_cv: int,
    results: dict,
) -> dict:
    """
    Perform train-test split analysis and save results to pickle files.

    References
    ----------
    1. Hastie T, Tibshirani R, Friedman JH. The elements of statistical learning:
       data mining, inference, and prediction. 2nd ed. Springer, 2009.
    2. Cawley GC, Talbot NLC. On Over-fitting in Model Selection and Subsequent
       Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : Dict
        Dictionary containing task configuration variables.
    g: pd.DataFrame
        DataFrame with group identifiers for cross-validation splits.
    x: pd.DataFrame
        DataFrame containing predictors.
    y: pd.DataFrame
        DataFrame containing target values.
    i_trn: list
        List of int indices for training.
    i_tst: list
        List of int indices for testing.
    i_cv: int
        Index of current repetition.
    results: dict
        Dictionary with results.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    # --- Initialize ---
    # Save start time
    t_start = time()
    # Get estimator
    estimator = get_estimator(task)

    # --- Subsample indices---
    # Subsample i_trn to max 10000 samples
    i_trn_sub = np.random.choice(
        i_trn,
        size=min(i_trn.shape[0], 10000),
        replace=False,
        p=None,
    )
    # Subsample i_tst to max 10000 samples
    i_tst_sub = np.random.choice(
        i_tst,
        size=min(i_tst.shape[0], 10000),
        replace=False,
        p=None,
    )

    # --- Split data ---
    # Split groups
    g_trn, g_tst = split_data(g, i_trn_sub, i_tst_sub)
    # Split targets
    y_trn, y_tst = split_data(y, i_trn_sub, i_tst_sub)
    # Split predictors
    x_trn, x_tst = split_data(x, i_trn_sub, i_tst_sub)

    # --- Fit, score, and explain ---
    # Fit estimator
    estimator.fit(x_trn, y_trn.squeeze())
    # Score predictions
    results["scores"].append(score_predictions(task, estimator, x_tst, y_tst, i_tst, y))
    # SHAP explanations
    results["explanations"].append(get_explanations(task, estimator, x_trn, x_tst))

    # --- Shuffle fit, score, and explain ---
    # Refit estimator with shuffled targets
    estimator_sh = estimator.fit(x_trn, shuffle(y_trn).squeeze())
    # # Score predictions
    results["scores_sh"].append(
        score_predictions(task, estimator_sh, x_tst, y_tst, i_tst, y)
    )
    # SHAP explanations
    results["explanations_sh"].append(
        get_explanations(task, estimator_sh, x_trn, x_tst)
    )

    # --- Save results and task configuration ---
    # Save results as pickle file
    with open(f"{task['save_path']}_results.pickle", "wb") as filehandle:
        # store the data as binary data stream
        pkl.dump(results, filehandle)
    # Save task as pickle file
    with open(f"{task['save_path']}_task.pickle", "wb") as filehandle:
        # store the data as binary data stream
        pkl.dump(task, filehandle)

    # --- Log current results ---
    log_current_results(
        task, t_start, results["scores"], results["scores_sh"], i_cv
    )

    # --- Add delay for storing data ---
    # If loop was faster than 1s
    if time() - t_start < 1:
        # Add a 1s sleep
        sleep(1)

    # --- return results ---
    return results


def run_modelling(
    task: dict, g: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame
) -> dict:
    """
    Perform modelling and save results to pickle files.

    References
    ----------
    1. Hastie T, Tibshirani R, Friedman JH. The elements of statistical learning:
       data mining, inference, and prediction. 2nd ed. Springer, 2009.
    2. Cawley GC, Talbot NLC. On Over-fitting in Model Selection and Subsequent
       Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : Dict
        Dictionary containing task configuration variables.
    g: pd.DataFrame
        DataFrame with group identifiers for cross-validation splits.
    x: pd.DataFrame
        DataFrame containing predictors.
    y: pd.DataFrame
        DataFrame containing target values.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Initialize results dict
    results = {
        "best_params": [],
        "best_pipe": [],
        "scores": [],
        "explanations": [],
        "scores_sh": [],
        "explanations_sh": [],
    }
    # If cross-validation
    if task["TYPE"] == "CV":
        # Choose n_rep_cv to approx N_PRED_CV (min 2).
        task["n_rep_cv"] = max(2, mth.ceil(task["N_PRED_CV"] / g.shape[0]))
        # Instatiate cv splitter
        cv = RepeatedGroupKFold(
            n_splits=task["N_CV_FOLDS"], n_repeats=task["n_rep_cv"], random_state=1000
        )
        # Loop over cv splits and repetitions
        for i_cv, (i_trn, i_tst) in enumerate(cv.split(g, groups=g)):
            # Run train test split
            single_train_test_split_predictions(
                task, g, x, y, i_trn, i_tst, i_cv, results
            )
    # If train-test split
    elif task["TYPE"] == "TT":
        # Get train data index
        i_trn = np.array(list(set(g.index).difference(set(task["TEST_SET_IND"]))))
        # Get test data index
        i_tst = np.array(task["TEST_SET_IND"])
        # Set n_rep_cv for compatibility
        task["n_rep_cv"] = 0
        # Set i_cv for compatibility
        i_cv = 0
        # Run train test split
        single_train_test_split_predictions(task, g, x, y, i_trn, i_tst, i_cv, results)
    # Other
    else:
        # Raise error
        raise ValueError(f"TYPE is {task['TYPE']}.")


def main() -> None:
    """
    Main function of Interpretable Machine-Learning.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    OSError: If create results directory failed.
    OSError: If copy iml_2_mdl script to results path failed.
    FileNotFoundError: If load G, X, or Y failed.
    OSError: If save g, x, or y failed.
    ValueError: If TYPE is not CV and TT.
    OSError: If copy log file to results directory failed.
    """

    ####################################################################################
    # Script Configuration
    ####################################################################################

    # --- Specify task ---

    # Type of analysis. str (default: CV, TT)
    TYPE = "CV"
    # Number parallel processing jobs. int (-1=all, default: -2=all-1)
    N_JOBS = -2
    # Number of folds in CV. int (default: 5)
    N_CV_FOLDS = 5
    # Number of predictions in 5 fold CV (if TYPE='CV'). int (default: 10000)
    # Be aware of hardcoded min. 2 repetition.
    N_PRED_CV = 10000
    # Number of samples SHAP. int (default: 1000).
    N_SAMPLES_SHAP = 1000
    # Device to run computations. str (default: "cuda", "cpu")
    DEVICE = "cuda"

    # --- Specify data ---

    # Concentration data - regression
    # Specifiy an analysis name
    ANALYSIS_NAME = "concentration"
    # Specify path to data. string
    PATH_TO_DATA = "sample_data/concentration_20250122.xlsx"
    # Specify sheet name. string
    SHEET_NAME = "data_nan"
    # Specify task OBJECTIVE. string (classification, regression)
    OBJECTIVE = "regression"
    # Specify grouping for CV split. list of string
    G_NAME = [
        "sample_id",
    ]
    # Specify predictor name(s). list of strings
    X_NAMES = [
        "chloride",
        "compound_8",
        "fluid_velocity",
        "nitrogen_nitrates",
        "nitrites_ammonia",
        "oxygen",
        "phosphate",
        "pH",
        "river_size",
        "season",
    ]
    # Specify indices for X_NAMES to target encode. list of int (default: [])
    TARGET_ENCODING_IND = []
    # Specify target name(s). list of strings
    Y_NAMES = [
        "concentration_a1",
        "concentration_a2",
    ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []
    # Specify index of rows for test set if TT. list of int or []
    TEST_SET_IND = list(randint.rvs(0, 199, size=40, random_state=1))

    # # Diabetes data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "diabetes"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/diabetes_20240806.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of strings
    # X_NAMES = [
    #     "age",
    #     "bmi",
    #     "bp",
    #     "gender",
    #     "s1_tc",
    #     "s2_ldl",
    #     "s3_hdl",
    #     "s4_tch",
    #     "s5_ltg",
    #     "s6_glu",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = []
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "progression",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 442, size=88, random_state=1000))

    # # Drug data - classification 5 class
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "drug"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/drug_20250116.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data_nan"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of strings
    # X_NAMES = [
    #     "age",
    #     "bp_lnh",
    #     "cholesterol_nh",
    #     "gender_fm",
    #     "na_to_k",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = []
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "drug",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 199, size=40, random_state=1000))

    # # Employee data - classification 2 class
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "employee"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/employee_20240806.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of strings
    # X_NAMES = [
    #     "age",
    #     "distance_from_home",
    #     "environment_satisfaction",
    #     "job_satisfaction",
    #     "monthly_income",
    #     "num_companies_worked",
    #     "stock_option_level",
    #     "training_times_last_year",
    #     "total_working_years",
    #     "work_life_balance",
    #     "years_at_company",
    #     "years_since_last_promotion",
    #     "years_with_curr_manager",
    #     "gender",
    #     "over_time",
    #     "marital_status",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = []
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "attrition",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 1470, size=294, random_state=1000))

    # # Housing data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "housing"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/housing_20240806.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of strings
    # X_NAMES = [
    #     "median_income",
    #     "house_age",
    #     "average_rooms",
    #     "average_bedrooms",
    #     "population",
    #     "average_occupation",
    #     "latitude",
    #     "longitude",
    #     "ocean_proximity",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = []
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "median_house_value",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 20640, size=4128, random_state=1000))

    # # Radon data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "radon"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/radon_20250116.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data_nan"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of strings
    # X_NAMES = [
    #     "uppm",
    #     "basement",
    #     "floor",
    #     "region",
    #     "room",
    #     "zip",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [5])
    # TARGET_ENCODING_IND = [5]
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "log_radon",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 918, size=184, random_state=1000))

    # # Wine data - classification 3 class
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "wine"
    # # Specify path to data. string
    # PATH_TO_DATA = "sample_data/wine_20240806.xlsx"
    # # Specify sheet name. string
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. string (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of string
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor names. list of strings
    # X_NAMES = [
    #     "alcohol",
    #     "malic_acid",
    #     "ash",
    #     "alcalinity_of_ash",
    #     "magnesium",
    #     "total_phenols",
    #     "flavanoids",
    #     "nonflavanoid_phenols",
    #     "proanthocyanins",
    #     "color_intensity",
    #     "hue",
    #     "od280_od315_of_diluted_wines",
    #     "proline",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = []
    # # Specify target name(s). list of strings
    # Y_NAMES = [
    #     "maker",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 178, size=36, random_state=1000))

    ####################################################################################

    # --- Configure logging ---
    # Basic configuration
    logging.basicConfig(
        filename=f"iml_2_mdl_tabpfn_{ANALYSIS_NAME}.log",  # Log file path
        filemode="w",  # Open the file in write mode to overwrite its content
        level=logging.INFO,  # Set the minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        force=True,
    )
    # Create a console handler for output to the terminal
    console_handler = logging.StreamHandler()
    # Define the console log format
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Set the console log format
    console_handler.setFormatter(console_formatter)
    # Set the console log level
    console_handler.setLevel(logging.INFO)
    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    logging.info(
        f"Interpretable Machine Learning - Modelling (MDL-tabpfn) of {ANALYSIS_NAME}."
    )

    # --- Create results directory ---
    # Create results path
    results_path = f"iml_2_mdl_tabpfn_{ANALYSIS_NAME}"
    try:
        # Create results directory
        os.makedirs(results_path, exist_ok=True)  # Supress FileExistsError
    except OSError as e:
        # Raise error
        raise e

    # --- Save pip requirements ---
    # Get pip requirements
    pip_requirements = get_pip_requirements()
    # Open file in write mode
    with open(f"{results_path}/iml_2_mdl_tabpfn_pip_requirements.txt", "w") as file:
        # Write pip requirements
        file.write(pip_requirements)

    # --- Save this python script ---
    try:
        # Copy iml_2_mdl script to results path
        shutil.copy("iml_2_mdl_tabpfn.py", f"{results_path}/iml_2_mdl_tabpfn.py")
    except OSError as e:
        # Raise error
        raise e

    # --- Create task dictionary ---
    task = {
        "TYPE": TYPE,
        "N_JOBS": N_JOBS,
        "N_CV_FOLDS": N_CV_FOLDS,
        "N_PRED_CV": N_PRED_CV,
        "N_SAMPLES_SHAP": N_SAMPLES_SHAP,
        "DEVICE": DEVICE,
        "ANALYSIS_NAME": ANALYSIS_NAME,
        "PATH_TO_DATA": PATH_TO_DATA,
        "SHEET_NAME": SHEET_NAME,
        "OBJECTIVE": OBJECTIVE,
        "G_NAME": G_NAME,
        "X_NAMES": X_NAMES,
        "TARGET_ENCODING_IND": TARGET_ENCODING_IND,
        "Y_NAMES": Y_NAMES,
        "SKIP_ROWS": SKIP_ROWS,
        "TEST_SET_IND": TEST_SET_IND,
        "results_path": results_path,
    }

    # --- Load data ---
    try:
        # Load groups data from excel file
        G = pd.read_excel(
            task["PATH_TO_DATA"],
            sheet_name=task["SHEET_NAME"],
            header=0,
            usecols=task["G_NAME"],
            dtype="int",
            skiprows=task["SKIP_ROWS"],
        )
    except FileNotFoundError as e:
        # Raise error
        raise e
    try:
        # Load predictors from excel file
        X = pd.read_excel(
            task["PATH_TO_DATA"],
            sheet_name=task["SHEET_NAME"],
            header=0,
            usecols=task["X_NAMES"],
            dtype="float",
            skiprows=task["SKIP_ROWS"],
        )
    except FileNotFoundError as e:
        # Raise error
        raise e
    # Reindex x to x_names
    X = X.reindex(task["X_NAMES"], axis=1)
    try:
        # Load targets from excel file
        Y = pd.read_excel(
            task["PATH_TO_DATA"],
            sheet_name=task["SHEET_NAME"],
            header=0,
            usecols=task["Y_NAMES"],
            dtype="float",
            skiprows=task["SKIP_ROWS"],
        )
    except FileNotFoundError as e:
        # Raise error
        raise e

    # --- Modelling ---
    # Iterate over prediction targets (Y_NAMES)
    for i_y, y_name in enumerate(Y_NAMES):
        # Log start modelling
        logging.info(f"Modelling started for target {y_name}.\n")
        # Add prediction target index to task
        task["i_y"] = i_y
        # Add prediction target name to task
        task["y_name"] = y_name
        # Make save path
        save_path = f"{task['results_path']}/iml_2_mdl_tabpfn_{task['y_name']}"
        # Add save path to task
        task["save_path"] = save_path

        # --- Deal with NaNs in the target ---
        # Get current target and remove NaNs
        y = Y[y_name].to_frame().dropna()
        # Use y index for groups and reset index
        g = G.reindex(index=y.index)
        # Use y index for predictors and reset index
        x = X.reindex(index=y.index)
        # If samples were dropped because of NaNs in target
        if y.shape[0] < Y.shape[0]:
            # Log warning
            logging.warning(
                f"{Y.shape[0]-y.shape[0]} samples dropped due to NaNs in {y_name}."
            )

        # --- Save data ---
        try:
            # Save group data
            g.to_excel(f"{task['save_path']}_data_g.xlsx", index=False)
        except OSError as e:
            # Raise error
            raise e
        try:
            # Save predictor data
            x.to_excel(f"{task['save_path']}_data_x.xlsx", index=False)
        except OSError as e:
            # Raise error
            raise e
        try:
            # Save target data
            y.to_excel(f"{task['save_path']}_data_y.xlsx", index=False)
        except OSError as e:
            # Raise error
            raise e

        # --- Get number of classes ---
        # If classification
        if task["OBJECTIVE"] == "classification":
            # Get number of classes in target
            task["n_classes"] = y.nunique()[task["y_name"]]
        # If regression
        elif task["OBJECTIVE"] == "regression":
            # Set number of classes to -1 for compatibility
            task["n_classes"] = -1
        else:
            # Raise error
            raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

        # --- Run modelling ---
        run_modelling(task, g, x, y)

    # --- Save log file to results directory ---
    # Log success
    logging.info(
        f"Interpretable Machine-Learning - Modelling (MDL) of {ANALYSIS_NAME} finished."
    )
    try:
        # Copy log file to results directory
        shutil.copy(
            f"iml_2_mdl_tabpfn_{ANALYSIS_NAME}.log",
            f"{results_path}/iml_2_mdl_tabpfn_{ANALYSIS_NAME}.log",
        )
    except OSError as e:
        # Raise error
        raise e
    # Stop logging
    logging.shutdown()
    # Delete the original log file
    os.remove(f"iml_2_mdl_tabpfn_{ANALYSIS_NAME}.log")


if __name__ == "__main__":
    main()
