# -*- coding: utf-8 -*-
"""
Interpretable Machine-Learning 2 - Modelling (MDL-lgbm)
v922
@author: david.steyrl@univie.ac.at
"""

import logging
import math as mth
import numpy as np
import os
import pandas as pd
import pickle as pkl
import shutil
import subprocess
import warnings
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from shap import Explanation
from shap.explainers import Tree as TreeExplainer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import shuffle
from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from time import sleep
from time import time

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
            "pip freeze",
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


def prepare_pipeline(task: dict) -> tuple:
    """
    Prepare analysis pipeline and search space.

    Parameters
    ----------
    task: dict
        Dictionary containing task details.

    Returns
    -------
    tuple: A tuple containing the prepared pipeline and search space.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Make preprocessing pipe ---
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
    # Pipeline
    preprocessing_pipeline = Pipeline(
        [("column_transformer", column_transformer), ("std_scaler", StandardScaler())],
        memory=None,
        verbose=False,
    ).set_output(transform="pandas")

    # --- Get predictor ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Predictor
        predictor = LGBMRegressor(
            boosting_type="gbdt",
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective="huber",
            min_split_gain=0.0,
            min_child_weight=0.00001,
            min_child_samples=1,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type="gain",
            **{
                "bagging_seed": None,
                "data_random_seed": None,
                "data_sample_strategy": "goss",
                "extra_seed": None,
                "feature_fraction_seed": None,
                "feature_pre_filter": False,
                "force_col_wise": True,
                "min_data_in_bin": 1,
                "verbosity": -1,
            },
        )
        # Add scaler to the predictor
        predictor = TransformedTargetRegressor(
            regressor=predictor,
            transformer=StandardScaler(),
            func=None,
            inverse_func=None,
            check_inverse=True,
        )
        # Search space
        space = {
            "predictor__regressor__colsample_bytree": uniform(0.1, 0.9),
            "predictor__regressor__extra_trees": [True, False],
            "predictor__regressor__reg_lambda": loguniform(0.1, 100),
        }
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Predictor
        predictor = LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective="multiclass",
            class_weight="balanced",
            min_split_gain=0.0,
            min_child_weight=0.00001,
            min_child_samples=1,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type="gain",
            **{
                "bagging_seed": None,
                "data_random_seed": None,
                "data_sample_strategy": "goss",
                "extra_seed": None,
                "feature_fraction_seed": None,
                "feature_pre_filter": False,
                "force_col_wise": True,
                "min_data_in_bin": 1,
                "num_class": task["n_classes"],
                "verbosity": -1,
            },
        )
        # Search space
        space = {
            "predictor__colsample_bytree": uniform(0.1, 0.9),
            "predictor__extra_trees": [True, False],
            "predictor__reg_lambda": loguniform(0.1, 100),
        }
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Make full pipeline ---
    # Analyis pipeline
    pipe = Pipeline(
        [("preprocessing_pipeline", preprocessing_pipeline), ("predictor", predictor)],
        memory=None,
        verbose=False,
    ).set_output(transform="pandas")

    # --- Return pipe and space ---
    return pipe, space


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


def log_tune_summary(task: dict, i_cv: int, hp_params: dict, hp_score: dict) -> None:
    """
    Log best hyper paramters and related scores.

    Parameters
    ----------
    task: dictionary
        Dictionary holding the task describtion variables.
    i_cv: int
        Current cv repetition.
    hp_params: dictionary
        Best hyper params found.
    hp_score: dictionary
        Score for best hyper params found.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    ValueError: If TYPE is not CV and TT.
    """
    # Log analysis name
    logging.info(f"Analysis: {task['ANALYSIS_NAME']}")
    # Log data set
    logging.info(f"Dataset: {task['PATH_TO_DATA']}")
    # Log prediction target
    logging.info(f"Predicting: {task['y_name']}")
    # If cross-validation
    if task["TYPE"] == "CV":
        # If regression
        if task["OBJECTIVE"] == "regression":
            # Log general information
            logging.info(
                f"{task['i_y']}.{i_cv} | n rep outer cv: {task['n_rep_outer_cv']} | n rep rs: {task['N_SAMPLES_RS']} | n rep inner cv: {task['n_rep_inner_cv']} | best R²: {np.round(hp_score, decimals=4)}"  # noqa
            )
        # If classification
        elif task["OBJECTIVE"] == "classification":
            # Log general information
            logging.info(
                f"{task['i_y']}.{i_cv} | n rep outer cv: {task['n_rep_outer_cv']} | n rep rs: {task['N_SAMPLES_RS']} | n rep inner cv: {task['n_rep_inner_cv']} | best acc: {np.round(hp_score, decimals=4)}"  # noqa
            )
        else:
            # Raise error
            raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # If Train-Test split
    elif task["TYPE"] == "TT":
        # Regression
        if task["OBJECTIVE"] == "regression":
            # Log general information
            logging.info(
                f"{task['i_y']}.{i_cv} | n rep rs: {task['N_SAMPLES_RS']} | n rep inner cv: {task['n_rep_inner_cv']} | best R²: {np.round(hp_score, decimals=4)}"  # noqa
            )
        # Classification
        elif task["OBJECTIVE"] == "classification":
            # Log general information
            logging.info(
                f"{task['i_y']}.{i_cv} | n rep rs: {task['N_SAMPLES_RS']} | n rep inner cv: {task['n_rep_inner_cv']} | best acc: {np.round(hp_score, decimals=4)}"  # noqa
            )
        else:
            # Raise error
            raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    else:
        # Raise error
        raise ValueError(f"TYPE is {task['TYPEE']}.")
    # Log best hyperparameter
    for key, value in hp_params.items():
        logging.info(f"Best hyperparameter {key}: {value}")


def optimize_pipeline(
    task: dict,
    i_cv: int,
    pipe: Pipeline,
    space: dict,
    g_trn: np.ndarray,
    x_trn: np.ndarray,
    y_trn: np.ndarray,
) -> tuple:
    """
    Performs hyperparameter optimization as part of the inner loop of nested
    cross-validation. Uses randomized search to identify optimal hyperparameters for
    the given pipeline, ensuring unbiased performance evaluation.

    References
    ----------
    - Hastie, T., Tibshirani, R., & Friedman, J.H. (2009). The Elements of Statistical
      Learning: Data Mining, Inference, and Prediction (2nd ed.). New York,
      NY: Springer.
    - Cawley, G.C., & Talbot, N.L.C. (2010). On Over-fitting in Model Selection and
      Subsequent Selection Bias in Performance Evaluation.

    Parameters
    ----------
    task: dict
        Dictionary containing task-specific configuration and metadata.
    i_cv : int
        Current iteration index for outer cross-validation.
    pipe: Pipeline
        A scikit-learn compatible analysis pipeline for preprocessing and model fitting.
    space : dict
        Hyperparameter search space specifying distributions or candidate values for
        tuning.
    g_trn: np.ndarray
        Array of group labels for the training data
        (used for group-aware cross-validation).
    x_trn: np.ndarray
        Feature matrix (n_samples, n_features) for training data.
    y_trn: np.ndarray
        Target vector (n_samples) for training data.

    Returns
    -------
    tuple: Containing pipe (the input pipeline fitted with the optimal hyperparameters
        on the training data) and dict (a dictionary of the best hyperparameters
        identified during the tuning process).

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Get scorer ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # R² score
        scorer = "r2"
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Balanced accuracy for classification
        scorer = "balanced_accuracy"
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Tune modelling pipeline ---
    # Choose n_rep_inner_cv to approx N_PRED_INNER_CV (min 2, max 5).
    task["n_rep_inner_cv"] = max(
        2, min(5, mth.ceil(task["N_PRED_INNER_CV"] / g_trn.shape[0]))
    )
    # Instatiate random parameter search
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=task["N_SAMPLES_RS"],
        scoring=scorer,
        n_jobs=task["N_JOBS"],
        refit=True,
        cv=RepeatedGroupKFold(
            n_splits=task["N_CV_FOLDS"],
            n_repeats=task["n_rep_inner_cv"],
            random_state=None,
        ),
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=0,
        return_train_score=False,
    )
    # Random search for best parameter
    search.fit(x_trn, y_trn.squeeze(), groups=g_trn)
    # Log tune summary
    log_tune_summary(task, i_cv, search.best_params_, search.best_score_)

    # --- Return tuned analysis pipeline ---
    return search.best_estimator_, search.best_params_


def score_predictions(
    task: dict,
    pipe: Pipeline,
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
    pipe: Pipeline
        A scikit-learn compatible pipeline used to generate predictions.
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

    # --- Predict ---
    # Predict test samples
    y_pred = pipe.predict(x_tst)

    # --- Score results ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {
            "x_tst": x_tst,
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "y_ind": i_tst,
            "mae": mae,
            "r2": r2,
        }
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Calculate model fit in terms of acc
        acc = balanced_accuracy_score(y_tst, y_pred)
        # Results
        scores = {
            "x_tst": x_tst,
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "y_ind": i_tst,
            "acc": acc,
        }
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Return scores ---
    return scores


def aggregate_shap_explanations(
    explanations: Explanation, features_to_group: list, new_feature_name: str
) -> Explanation:
    """
    Aggregates the SHAP values of a given list of features into a new aggregated
    feature, using the additivity property of SHAP values.

    Parameters
    ----------
    explanations: SHAP explanation object
        Assumed to have attributes:
          - values: a numpy array of shape (n_samples, n_features, n_classes)
            or (n_samples, n_features, n_features, n_classes) for interaction values.
          - feature_names: list of feature names.
    features_to_group: list of str
        List of feature names to aggregate.
    new_feature_name: str
        Name for the aggregated feature.

    Returns
    -------
    Explanation: SHAP explanation object
        A new explanation object with the specified features aggregated.

    Raises
    ------
    ValueError: If SHAP values array ndim>4.
    """

    # --- Prepare ---
    # Get original feature names
    orig_feature_names = explanations.feature_names
    # Find indices of features to group
    group_indices = [orig_feature_names.index(f) for f in features_to_group]
    # Determine remaining feature indices
    remaining_indices = [
        i for i in range(len(orig_feature_names)) if i not in group_indices
    ]
    # Create new feature names list: keep remaining in orig order, then add new feature
    new_feature_names = [orig_feature_names[i] for i in remaining_indices] + [
        new_feature_name
    ]
    # Get shap values, could be 3D or 4D
    shap_vals = explanations.values
    # Get shap values dimensions
    ndim = shap_vals.ndim

    # --- Sum SHAP values ---
    # If N dim is 3 (n_samples, n_features, n_classes)
    if ndim == 3:
        # Aggregate SHAP values: sum over group features. shape (n_samples, n_classes)
        aggregated_vals = np.sum(shap_vals[:, group_indices, :], axis=1)
        # Retain SHAP values for remaining. shape (n_samples, len(remaining), n_classes)
        remaining_vals = shap_vals[:, remaining_indices, :]
        # Concatenate along feature axis.
        new_vals = np.concatenate(
            [remaining_vals, aggregated_vals[:, np.newaxis, :]], axis=1
        )
    # If N dim is 4 (n_samples, n_features, n_features, n_classes)
    elif ndim == 4:
        # Get dimensions of the explanation object
        n_samples, M, _, n_classes = shap_vals.shape
        # New number of features after aggregation
        new_M = len(remaining_indices) + 1
        # Initialize new array.
        new_vals = np.zeros((n_samples, new_M, new_M, n_classes))
        # Mapping from original index to new index for remaining features.
        mapping = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(remaining_indices)
        }
        # Fill interactions for remaining features (i, j both not in group)
        for i in remaining_indices:
            for j in remaining_indices:
                new_i = mapping[i]
                new_j = mapping[j]
                new_vals[:, new_i, new_j, :] = shap_vals[:, i, j, :]
        # New aggregated feature index
        agg_idx = new_M - 1
        # For interactions between aggregated feature and any remaining feature j
        for j in remaining_indices:
            # Map index
            new_j = mapping[j]
            # Row for aggregated feature: sum over all group features
            new_vals[:, agg_idx, new_j, :] = np.sum(
                shap_vals[:, group_indices, j, :], axis=1
            )
            # Column for aggregated feature: sum over all group features
            new_vals[:, new_j, agg_idx, :] = np.sum(
                shap_vals[:, j, group_indices, :], axis=1
            )
        # For the aggregated feature with itself: sum over all pairs within the group
        new_vals[:, agg_idx, agg_idx, :] = np.sum(
            shap_vals[:, group_indices, :][:, :, group_indices, :], axis=(1, 2)
        )
    else:
        # Raise error
        raise ValueError(f"Unsupported SHAP values array with ndim={ndim}")

    # --- Return explainer object ---
    return Explanation(
        new_vals,
        base_values=explanations.base_values,
        data=explanations.data,
        display_data=None,
        instance_names=None,
        feature_names=new_feature_names,
        output_names=None,
        output_indexes=None,
        lower_bounds=None,
        upper_bounds=None,
        error_std=None,
        main_effects=None,
        hierarchical_values=None,
        clustering=None,
        compute_time=None,
    )


def get_explanations(
    task: dict,
    pipe: Pipeline,
    x_tst: np.ndarray,
    y_tst: np.ndarray,
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
    pipe: Pipeline
        Fitted scikit-learn compatible pipeline with tuned parameters for generating
        predictions.
    x_tst: np.ndarray
        Test dataset (n_samples, n_features) on which SHAP values are computed to
        explain the model's predictions.
    y_tst: np.ndarray
        Test labels (n_samples, n_features) included in explainer object to be stored
        along with x_tst_shap.

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
            mth.ceil(
                task["N_SAMPLES_SHAP"] / (task["n_rep_outer_cv"] * task["N_CV_FOLDS"])
            ),
        )
    # If TT
    elif task["TYPE"] == "TT":
        # Get max samples shap
        task["max_samples_shap"] = min(x_tst.shape[0], task["N_SAMPLES_SHAP"])
    else:
        # Raise error
        raise ValueError(f"TYPE is {task['TYPE']}.")
    # Subsample test data
    x_tst_shap_orig = x_tst.sample(n=task["max_samples_shap"], random_state=1000)
    # Slice targets to fit subsampled predictors
    y_tst_shap = y_tst.loc[x_tst_shap_orig.index, :].reset_index(drop=True)
    # Reset index of predictors
    x_tst_shap_orig = x_tst_shap_orig.reset_index(drop=True)
    # Transform shap test data
    x_tst_shap = pipe[0].transform(x_tst_shap_orig)

    # --- Explainer and Explanations ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Get predictor
        predictor = pipe[1].regressor_
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Get predictor
        predictor = pipe[1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Get explainer
    explainer = TreeExplainer(
        predictor,
        data=None,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=None,
        approximate=False,
    )
    # Get explanations
    explanations = explainer(x_tst_shap, interactions=True, check_additivity=False)

    # --- Post process shap_explanations ---
    # If regression, undo target transformation effect on base values and shap values
    if task["OBJECTIVE"] == "regression":
        # Get target transformer scale from pipeline
        tf_scale = pipe["predictor"].transformer_.scale_[0]
        # Get target transformer mean from pipeline
        tf_mean = pipe["predictor"].transformer_.mean_[0]
        # Rescale shap base values from transformed target to original space
        explanations.base_values = (explanations.base_values * tf_scale) + tf_mean
        # Rescale shap values from scaled data to original space
        explanations.values = explanations.values * tf_scale
    # If multiclass, aggregate multi categorical predictor shap values
    if task["OBJECTIVE"] == "classification" and task["n_classes"] > 2:
        # If multi categorical predictors
        if task["TARGET_ENCODING_IND"]:
            # Loop over multi categorical predictors
            for name in [task["X_NAMES"][i] for i in task["TARGET_ENCODING_IND"]]:
                # Get names of variables to group
                group = [n for n in explanations.feature_names if name in n]
                # Aggregate shap values
                explanations = aggregate_shap_explanations(
                    explanations=explanations,
                    features_to_group=group,
                    new_feature_name=name,
                )
    # Replace scaled data in explanations with original
    explanations.data = x_tst_shap_orig[explanations.feature_names]
    # Add labels to explanations
    explanations.labels = y_tst_shap

    # --- Return explanations ---
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
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Log current R2
        logging.info(
            f"Current CV loop R²: {np.round(scores[-1]['r2'], decimals=4)}"
        )  # noqa
        # Log running mean R2
        logging.info(
            f"Running mean R²: {np.round(np.nanmean([i['r2'] for i in scores]), decimals=4)}"  # noqa
        )
        # Log running mean shuffle R2
        logging.info(
            f"Running shuffle mean R²: {np.round(np.nanmean([i['r2'] for i in scores_sh]), decimals=4)}"  # noqa
        )
        # Log elapsed time
        logging.info(f"Elapsed time: {np.round(time() - t_start, decimals=1)}\n")
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Log current acc
        logging.info(f"Current CV loop acc: {np.round(scores[-1]['acc'], decimals=4)}")
        # Log running mean acc
        logging.info(
            f"Running mean acc: {np.round(np.nanmean([i['acc'] for i in scores]), decimals=4)}"  # noqa
        )
        # Log running mean shuffle acc
        logging.info(
            f"Running shuffle mean acc: {np.round(np.nanmean([i['acc'] for i in scores_sh]), decimals=4)}"  # noqa
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
    # Prepare pipeline and search space
    pipe, space = prepare_pipeline(task)

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

    # --- Tune and fit ---
    # Get optimized and fitted pipe
    best_pipe, best_params = optimize_pipeline(
        task, i_cv, pipe, space, g_trn, x_trn, y_trn
    )
    # Store best params
    results["best_params"].append(best_params)
    # Store best pipe
    results["best_pipe"].append(best_pipe)

    # --- Get predictions and explanations ---
    # Score predictions
    results["scores"].append(score_predictions(task, best_pipe, x_tst, y_tst, i_tst, y))
    # SHAP explanations
    results["explanations"].append(
        get_explanations(task, best_pipe, x_tst, y_tst)
    )

    # --- Get shuffle predictions and explanations ---

    # Refit pipe with shuffled targets
    pipe_sh = best_pipe.fit(x_trn, shuffle(y_trn).squeeze())
    # Score predictions
    results["scores_sh"].append(
        score_predictions(task, pipe_sh, x_tst, y_tst, i_tst, y)
    )
    # SHAP explanations
    results["explanations_sh"].append(
        get_explanations(task, pipe_sh, x_tst, y_tst)
    )

    # --- Save intermediate results and task configuration ---
    # Save results as pickle file
    with open(f"{task['save_path']}_results.pickle", "wb") as filehandle:
        # store the data as binary data stream
        pkl.dump(results, filehandle)
    # Save task as pickle file
    with open(f"{task['save_path']}_task.pickle", "wb") as filehandle:
        # store the data as binary data stream
        pkl.dump(task, filehandle)

    # --- Log current results ---
    log_current_results(task, t_start, results["scores"], results["scores_sh"], i_cv)

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
        # Choose n_rep_outer_cv to approx N_PRED_OUTER_CV (min 2).
        task["n_rep_outer_cv"] = max(2, mth.ceil(task["N_PRED_OUTER_CV"] / g.shape[0]))
        # Instatiate cv splitter
        cv = RepeatedGroupKFold(
            n_splits=task["N_CV_FOLDS"],
            n_repeats=task["n_rep_outer_cv"],
            random_state=1000,
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
    OSError: If create store directory failed.
    OSError: If save pip requirements to store failed.
    OSError: If copy iml_2_mdl_lgbm script to store failed.
    FileNotFoundError: If load G, X, or Y to store failed.
    OSError: If save g, x, or y failed.
    ValueError: If OBJECTIVE is not regression and classification.
    ValueError: If TYPE is not CV and TT.
    OSError: If copy log file to store failed.
    """

    ####################################################################################
    # Script Configuration
    ####################################################################################

    # --- Specify task ---

    # Type of analysis. str (default: CV, TT)
    TYPE = "CV"
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    N_JOBS = -2
    # Number of folds in CV. int (default: 5)
    N_CV_FOLDS = 5
    # Number of predictions in outer 5-fold CV (if TYPE='CV'). int (default: 10000)
    # Be aware of hardcoded min. 2 repetitions.
    N_PRED_OUTER_CV = 10000
    # Number of attempts in random search. int (default: 100)
    N_SAMPLES_RS = 100
    # Number of predictions in inner 5-fold CV. int (default: 1000)
    # Be aware of hardcoded min. 2 and max 5 repetitions.
    N_PRED_INNER_CV = 1000
    # Number of samples SHAP. int (default: 1000).
    N_SAMPLES_SHAP = 1000
    # Store prefix (where results go). str
    STORE_PREFIX = "iml_2_mdl_lgbm_"

    # --- Specify data ---

    # Concentration data - regression
    # Specifiy an analysis name
    ANALYSIS_NAME = "concentration"
    # Specify path to data. str
    PATH_TO_DATA = "sample_data/concentration_20250122.xlsx"
    # Specify sheet name. str
    SHEET_NAME = "data_nan"
    # Specify task OBJECTIVE. str (classification, regression)
    OBJECTIVE = "regression"
    # Specify grouping for CV split. list of str
    G_NAME = [
        "sample_id",
    ]
    # Specify predictor name(s). list of str
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
    # Specify target name(s). list of str
    Y_NAMES = [
        "concentration_a1",
        "concentration_a2",
    ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []
    # Specify index of rows for test set if TT. list of int or []
    TEST_SET_IND = list(randint.rvs(0, 199, size=40, random_state=1000))

    # # Diabetes data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "diabetes"
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/diabetes_20240806.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of str
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
    # # Specify target name(s). list of str
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
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/drug_20250116.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data_nan"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of str
    # X_NAMES = [
    #     "age",
    #     "bp_lnh",
    #     "cholesterol_nh",
    #     "gender_fm",
    #     "na_to_k",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = [1]
    # # Specify target name(s). list of str
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
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/employee_20240806.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of str
    # X_NAMES = [
    #     "age",
    #     "gender",
    #     "marital_status",
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
    #     "over_time",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = [2]
    # # Specify target name(s). list of str
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
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/housing_20240806.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of str
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
    # # Specify target name(s). list of str
    # Y_NAMES = [
    #     "median_house_value",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 20640, size=4128, random_state=1000))

    # # # Radon data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = "radon"
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/radon_20250116.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data_nan"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "regression"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor name(s). list of str
    # X_NAMES = [
    #     "uppm",
    #     "basement",
    #     "floor",
    #     "room",
    #     "zip",
    #     "region",
    # ]
    # # Specify indices for X_NAMES to target encode. list of int (default: [])
    # TARGET_ENCODING_IND = [4]
    # # Specify target name(s). list of str
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
    # # Specify path to data. str
    # PATH_TO_DATA = "sample_data/wine_20240806.xlsx"
    # # Specify sheet name. str
    # SHEET_NAME = "data"
    # # Specify task OBJECTIVE. str (classification, regression)
    # OBJECTIVE = "classification"
    # # Specify grouping for CV split. list of str
    # G_NAME = [
    #     "sample_id",
    # ]
    # # Specify predictor names. list of str
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
    # TARGET_ENCODING_IND = [4]
    # # Specify target name(s). list of str
    # Y_NAMES = [
    #     "maker",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []
    # # Specify index of rows for test set if TT. list of int or []
    # TEST_SET_IND = list(randint.rvs(0, 178, size=36, random_state=1000))

    ####################################################################################

    # --- Configure logging ---
    # Make log filename
    log_filename = f"{STORE_PREFIX}{ANALYSIS_NAME}.log"
    # Basic configuration
    logging.basicConfig(
        # Log file path
        filename=log_filename,
        # Open the file in write mode to overwrite its content
        filemode="w",
        # Set the minimum log level
        level=logging.INFO,
        # Log format
        format="%(asctime)s - %(levelname)s - %(message)s",
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
        f"Interpretable Machine Learning - Modelling (MDL-lgbm) of {ANALYSIS_NAME} started."  # noqa
    )

    # --- Store directory ---
    # Make store path (where results go)
    store_path = f"{STORE_PREFIX}{ANALYSIS_NAME}"
    try:
        # Create results directory
        os.makedirs(store_path, exist_ok=True)  # Supress FileExistsError
    except OSError as e:
        # Raise error
        raise e

    # --- Pip requirements ---
    try:
        # Get pip requirements
        pip_requirements = get_pip_requirements()
        # Open file in write mode
        with open(f"{store_path}/{STORE_PREFIX}pip_requirements.txt", "w") as file:
            # Write pip requirements
            file.write(pip_requirements)
    except OSError as e:
        # Raise error
        raise e

    # --- Python script ---
    try:
        # Copy iml_2_mdl script to store path
        shutil.copy("iml_2_mdl_lgbm.py", f"{store_path}/iml_2_mdl_lgbm.py")
    except OSError as e:
        # Raise error
        raise e

    # --- Task dictionary ---
    task = {
        "TYPE": TYPE,
        "N_JOBS": N_JOBS,
        "N_CV_FOLDS": N_CV_FOLDS,
        "N_PRED_OUTER_CV": N_PRED_OUTER_CV,
        "N_PRED_INNER_CV": N_PRED_INNER_CV,
        "N_SAMPLES_RS": N_SAMPLES_RS,
        "N_SAMPLES_SHAP": N_SAMPLES_SHAP,
        "STORE_PREFIX": STORE_PREFIX,
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
        "store_path": store_path,
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
    # Reindex x to X_NAMES
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
        save_path = f"{task['store_path']}/{STORE_PREFIX}{task['y_name']}"
        # Add save path to task
        task["save_path"] = save_path

        # --- NaNs in target ---
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

        # --- Num classes ---
        # If classification
        if task["OBJECTIVE"] == "classification":
            # Get number of unique classes in prediction target
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

    # --- Save log file ---
    # Log success
    logging.info(
        f"Interpretable Machine Learning - Modelling (MDL-lgbm) of {ANALYSIS_NAME} finished."  # noqa
    )
    try:
        # Copy log file to store directory
        shutil.copy(
            log_filename,
            f"{store_path}/{log_filename}",
        )
    except OSError as e:
        # Raise error
        raise e
    # Stop logging
    logging.shutdown()
    # Delete the original log file
    os.remove(log_filename)


if __name__ == "__main__":
    main()
