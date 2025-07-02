# -*- coding: utf-8 -*-
"""
Interpretable Machine-Learning 1 - Exploratory Data Analysis (EDA)
v261
@author: david.steyrl@univie.ac.at
"""

import logging
import math as mth
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import subprocess
import warnings
from itertools import permutations
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
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


def get_estimator(
    task: dict, objective: str
) -> Union[TabPFNRegressor, TabPFNClassifier]:
    """
    Prepare analysis pipeline and search space.

    Parameters
    ----------
    task: dict
        Dictionary containing task details.
    objective: str
        Current objective (classification or regression).

    Returns
    -------
    Union[TabPFNRegressor, TabPFNClassifier]: A tuple containing the prepared pipeline
        and search space.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # If regression
    if objective == "regression":
        # Get estimator
        estimator = TabPFNRegressor(
            n_estimators=1,
            categorical_features_indices=None,
            softmax_temperature=0.9,
            average_before_softmax=False,
            device=task["DEVICE"],
            n_jobs=-2,
        )
    # If classification
    elif objective == "classification":
        # Get estimator
        estimator = TabPFNClassifier(
            n_estimators=1,
            categorical_features_indices=None,
            softmax_temperature=0.9,
            balance_probabilities=True,
            average_before_softmax=False,
            device=task["DEVICE"],
            n_jobs=-2,
        )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
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
    df_trn = df.iloc[i_trn].reset_index(drop=True)
    # Perform the split for testing
    df_tst = df.iloc[i_tst].reset_index(drop=True)
    # Return train test dataframes
    return df_trn, df_tst


def compute_pair_predictions(
    task: dict, g: pd.Series, x: pd.Series, y: pd.Series, objective: str
) -> float:
    """
    Compute pairwise prediction score
    (R² for regression, adjusted balanced accuracy for classification).

    Parameters
    ----------
    task: dict
        Dictionary holding task configuration parameters.
    g: pd.Series
        Series holding group data for cross-validation.
    x: pd.Series
        Series holding the predictor data.
    y: pd.Series
        Series holding the target data.
    objective: str
        Objective type, either "regression" or "classification".

    Returns
    -------
    float: Pairwise prediction score (≥0). R² for regression or adjusted balanced
        accuracy for classification.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Setup ---
    # Identify rows with NaN in either DataFrame
    nan_mask = x.isna().any(axis=1) | y.isna().any(axis=1)
    # Drop rows with NaN
    g = g[~nan_mask]
    x = x[~nan_mask]
    y = y[~nan_mask]
    # Choose n_rep_cv to approx N_PRED_CV (min 2, max 5 reps).
    task["n_rep_cv"] = max(2, min(5, mth.ceil(task["N_PRED_CV"] / g.shape[0])))
    # Instatiate cv splitter
    cv = RepeatedGroupKFold(
        n_splits=task["N_CV_FOLDS"],
        n_repeats=task["n_rep_cv"],
        random_state=None,
    )
    # Initialize scores
    scores = []
    # Get estimator
    estimator = get_estimator(task, objective)

    # --- Main CV loop ---
    for i_cv, (i_trn, i_tst) in enumerate(cv.split(g, groups=g)):

        # --- Split data ---
        # Split groups
        g_trn, g_tst = split_data(g, i_trn, i_tst)
        # Split targets
        y_trn, y_tst = split_data(y, i_trn, i_tst)
        # Split predictors
        x_trn, x_tst = split_data(x, i_trn, i_tst)

        # --- Fit, predict, and score ---
        # Fit model
        estimator.fit(x_trn, y_trn.squeeze())
        # Predict test samples
        y_pred = estimator.predict(x_tst)
        # Compute prediction score
        if objective == "classification":
            # Calculate model fit in terms of balanced acc
            scores.append(balanced_accuracy_score(y_tst, y_pred, adjusted=True))
        elif objective == "regression":
            # Score predictions in terms of R²
            scores.append(r2_score(y_tst, y_pred))
        else:
            # Raise error
            raise ValueError(f"Objective is {objective}.")

    # --- Final score processing ---
    # Limit pairwise predictions scores to be bigger than or equal to 0
    pair_pred = max(0, np.mean(scores))

    # --- Return pairwise predictions ---
    return pair_pred


def eda(task: dict, g: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> None:
    """
    Carries out exploratory data analysis, incl.:
    Data distribuations (1D, violinplot),
    Data distributions (2D, pairplots),
    Data joint information (linear, heatmap),
    Data joint information (non-linear, heatmap),
    Multidimensional pattern in data via PCA (linear, heatmap),
    Outlier in data (non-linear, histogram).

    Parameters
    ----------
    task: dictionary
        Dictionary holding the task describtion variables.
    g: dataframe
        Dataframe holding the group data.
    x: dataframe
        Dataframe holding the predictor data.
    y: dataframe
        Dataframe holding the target data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- 1D Data Distributions ---
    logging.info("1D Data Distributions.")
    # If DATA_DISTRIBUTION_1D
    if task["DATA_DISTRIBUTION_1D"]:
        # Step 1: Compute dimensions for the figure
        # Max length of feature names
        x_names_max_len = max(len(name) for name in task["X_NAMES"])
        # Total number of features
        x_names_count = len(task["X_NAMES"])
        # Compute fig width
        fig_size = (
            x_names_max_len * 0.1 + 5,
            (x_names_count + 1) * 1.1 + 1,
        )
        # Step 2: Create the figure and axes
        fig, axes = plt.subplots(
            nrows=x_names_count + 1, ncols=1, figsize=fig_size, sharex=False
        )
        # Step 3: Generate the violin plot
        for ax, column in zip(axes, pd.concat([x, y], axis=1).columns):
            sns.violinplot(
                data=pd.concat([x, y], axis=1)[column],
                bw_method="scott",  # Bandwidth estimation method
                bw_adjust=0.33,  # Adjust bandwidth
                cut=2,  # Extend density beyond data
                density_norm="width",  # Normalize density by width
                gridsize=100,  # Number of points in density estimation
                width=0.8,  # Width of violin plot
                inner="box",  # Show box plot inside violins
                orient="h",  # Horizontal orientation
                linewidth=1,  # Line width of violin edges
                color="#777777",  # Violin color
                saturation=1.0,  # Saturation level
                ax=ax,  # Axes to plot on
            )
            # Remove frame elements
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            ax.set_ylabel(
                column, rotation="horizontal", horizontalalignment="right", fontsize=10
            )
            # Set axis label
            ax.set_xlabel("", fontsize=10)
            # Add gridlines for better readability
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        # Set axis label
        ax.set_xlabel("Range", fontsize=10)
        # Step 4: Set the plot title
        title_str = f"{task['ANALYSIS_NAME']}\nData Distributions (1D)"
        plt.suptitle(title_str, fontsize=10, y=0.94)
        # Call tight layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Step 5: Save the figure
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_1_distri_1D"  # noqa
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # If as SVG
        if task["AS_SVG"]:
            # Make SVG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            plt.savefig(svg_save_path, bbox_inches="tight")

        # Step 6: Display the plot
        plt.show()

    # --- 2D Data Distributions ---
    logging.info("2D Data Distributions.")
    # If DATA_DISTRIBUTION_2D
    if task["DATA_DISTRIBUTION_2D"]:
        # Step 1: Create pairplot
        pair_plot = sns.pairplot(
            pd.concat([x, y], axis=1),
            corner=False,  # Include all pairwise plots
            diag_kind="kde",  # Use Kernel Density Estimate (KDE) for diagonal
            plot_kws={"color": "#777777"},  # Set color for pairwise plots
            diag_kws={"color": "#777777"},  # Set color for diagonal plots
        )
        # Step 2: Set plot title
        # Make title string
        title_str = f"{task['ANALYSIS_NAME']}\nData Distributions (2D)\n"
        # Set plot title
        pair_plot.fig.suptitle(title_str, fontsize=10, y=1.0)
        # Step 3: Add KDE plots for lower triangle
        pair_plot.map_lower(sns.kdeplot, levels=3, color=".2")
        # Step 4: Save the figure
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_2_distri_2D"  # noqa
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        pair_plot.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # If as SVG
        if task["AS_SVG"]:
            # Make SVG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            pair_plot.savefig(svg_save_path, bbox_inches="tight")
        # Step 5: Display the pairplot
        plt.show()

    # --- Linear Joint Information (Correlation Heatmap) ---
    logging.info("Linear Joint Information (Correlation Heatmap).")
    # If DATA_JOINT_INFORMATION_LINEAR
    if task["DATA_JOINT_INFORMATION_LINEAR"]:
        # Step 1: Compute figure size based on feature count and name lengths
        # Max length of feature names
        x_names_max_len = max(len(name) for name in task["X_NAMES"])
        # Total number of features
        x_names_count = len(task["X_NAMES"])
        # Compute fig size
        fig_size = (
            x_names_count * 0.6 + x_names_max_len * 0.1 + 1,
            x_names_count * 0.6 + x_names_max_len * 0.1 + 1,
        )
        # Step 2: Create heatmap figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        colorbar_label = "Correlation (0 to 1)"
        # Step 3: Generate correlation heatmap
        sns.heatmap(
            pd.concat([x, y], axis=1).corr().abs(),  # Compute correlations
            vmin=0,
            vmax=1,  # Value range
            cmap="Greys",  # Color map
            center=None,
            robust=True,
            annot=True,  # Show correlation values
            fmt=".2f",  # Format for annotations
            annot_kws={"size": 10},  # Annotation font size
            linewidths=1,  # Grid line width
            linecolor="#999999",  # Grid line color
            cbar=True,  # Show color bar
            cbar_kws={"label": colorbar_label, "shrink": 0.6},  # Color bar settings
            square=True,  # Square cells
            xticklabels=pd.concat([x, y], axis=1).columns,
            yticklabels=pd.concat([x, y], axis=1).columns,
            ax=ax,
        )
        # Step 4: Customize plot
        # Rotate axis labels
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        # Set title
        title_str = f"{task['ANALYSIS_NAME']}\nJoint Information in Data (Linear, Correlation)\n"  # noqa
        plt.title(title_str, fontsize=10)
        # Modify colorbar appearance
        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=10)
        cb_ax.set_ylabel(colorbar_label, fontsize=10)
        cb_ax.set_box_aspect(50)
        # Step 5: Save the figure
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_3_joint_lin"  # noqa
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # If as SVG
        if task["AS_SVG"]:
            # Make SVG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            plt.savefig(svg_save_path, bbox_inches="tight")
        # Step 6: Display the heatmap
        plt.show()

    # --- Non-linear Joint Information (Pairwise Prediction Heatmap) ---
    logging.info("Non-linear Joint Information (Pairwise Prediction Heatmap).")
    # If DATA_JOINT_INFORMATION_NON_LINEAR
    if task["DATA_JOINT_INFORMATION_NON_LINEAR"]:
        # Step1: Initialize pairwise prediction matrix
        # Get feature count
        feature_count = len(pd.concat([x, y], axis=1).columns)
        # Get pair predition matrix initialized by ones
        pair_predictions = np.ones((feature_count, feature_count))
        # Step2: Create pairwise combinations of features
        feature_indices = pd.factorize(pd.Series(pd.concat([x, y], axis=1).columns))[0]
        # Map indices to feature names
        mapping = list(pd.concat([x, y], axis=1).columns)
        # Step3: Iterate over pairwise combinations of features
        for id_pred1, id_pred2 in permutations(feature_indices, 2):
            # Get current predictor name
            predictor_name = mapping[id_pred1]
            # Get current target name
            target_name = mapping[id_pred2]
            # Step4: Prepare predictor (xt) and target (yt)
            # Get current predictor data
            xt = pd.DataFrame(pd.concat([x, y], axis=1)[predictor_name])
            # Get current target data
            yt = pd.DataFrame(pd.concat([x, y], axis=1)[target_name])
            # Step5: Determine the objective
            # If target_name in task["Y_NAMES"]
            if target_name in task["Y_NAMES"]:
                # Set current objective
                objective = task["OBJECTIVE"]
            # If target_name in task["X_CON_NAMES"]
            elif target_name in task["X_NAMES"]:
                # If 10 or less unique instances, treat as classification
                if (
                    pd.DataFrame(pd.concat([x, y], axis=1)[target_name])
                    .nunique()
                    .iloc[0]
                    .item()
                    <= 10
                ):
                    # Set current objective
                    objective = "classification"
                else:
                    # Set current objective
                    objective = "regression"
            else:
                raise ValueError("OBJECTIVE not found.")
            # Step6: Compute pairwise prediction for the current pair
            pair_predictions[id_pred1, id_pred2] = compute_pair_predictions(
                task=task, g=g, x=xt, y=yt, objective=objective
            )
        # Step7: Compute figure dimensions based on feature names
        max_name_length = max(len(name) for name in pd.concat([x, y], axis=1).columns)
        fig_size = (
            feature_count * 0.6 + max_name_length * 0.1 + 1,
            feature_count * 0.6 + max_name_length * 0.1 + 1,
        )
        # Step8: Create heatmap figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        colorbar_label = "Joint Information (0 to 1)"
        # Step9: Make heatmap
        sns.heatmap(
            pair_predictions,
            vmin=0,
            vmax=1,  # Value range
            cmap="Greys",  # Color map
            center=None,
            robust=True,
            annot=True,  # Show correlation values
            fmt=".2f",  # Format for annotations
            annot_kws={"size": 10},  # Annotation font size
            linewidths=1,  # Grid line width
            linecolor="#999999",  # Grid line color
            cbar=True,  # Show color bar
            cbar_kws={"label": colorbar_label, "shrink": 0.6},  # Color bar settings
            square=True,  # Square cells
            xticklabels=pd.concat([x, y], axis=1).columns,
            yticklabels=pd.concat([x, y], axis=1).columns,
            ax=ax,
        )
        # Step10: Set heatmap title
        title_str = (
            f"{task['ANALYSIS_NAME']}\n"
            "Joint Information in Data (Non-Linear, Pairwise Predictions)\n"
            "Y-axis: Predictors, X-axis: Prediction Targets"
        )
        plt.title(title_str, fontsize=10)
        # Step11: Save heatmap figure
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_4_joint_nonlin"  # noqa
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # Optionally save as SVG
        if task["AS_SVG"]:
            # Make SVG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            plt.savefig(svg_save_path, bbox_inches="tight")
        # Step12: Display the heatmap
        plt.show()

    # --- Multidimensional Pattern Visualization with PCA ---
    logging.info("Multidimensional Pattern Visualization with PCA.")
    # If DATA_MULTIDIM_PATTERN
    if task["DATA_MULTIDIM_PATTERN"]:
        # Step1: Instantiate PCA
        pca = PCA(
            n_components=x.shape[1],
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0001,
            iterated_power="auto",
            random_state=None,
        )
        # Step2: Fit PCA to the data
        pca.fit(x.dropna())
        # Step3: Extract number of features
        x_names_count = len(task["X_NAMES"])
        # Step4: Calculate figure size
        fig_width = min((1 + x_names_count * 0.6), 16)
        fig_size = (fig_width, 4)
        # Step5: Create the figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        # Step6: Plot explained variance ratio
        ax.plot(
            pca.explained_variance_ratio_,
            color="cornflowerblue",
            label="Explained variance per component",
        )
        ax.plot(
            pca.explained_variance_ratio_,
            color="black",
            marker=".",
            linestyle="None",
        )
        # Step7: Set limits for better visualization
        ax.set_xlim((-0.01, ax.get_xlim()[1]))
        ax.set_ylim((-0.01, 1.01))
        # Step8: Format the main axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("PCA-component")
        ax.set_ylabel("Explained Variance", color="cornflowerblue")
        # Step9: Create a twin y-axis for cumulative variance
        ax2 = ax.twinx()
        ax2.plot(
            np.cumsum(pca.explained_variance_ratio_),
            color="orange",
            label="Cumulative explained variance",
        )
        ax2.plot(
            np.cumsum(pca.explained_variance_ratio_),
            color="black",
            marker=".",
            linestyle="None",
        )
        # Step10: Set limits for the twin axis
        ax2.set_xlim((-0.01, ax2.get_xlim()[1]))
        ax2.set_ylim((-0.01, 1.01))
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.set_ylabel("Cumulative Variance", color="orange")
        # Step11: Add labels for explained variance and cumulative variance
        for comp, t in enumerate(pca.explained_variance_ratio_.round(2)):
            ax.text(comp, t, t, fontsize=10)
        for comp, t in enumerate(np.cumsum(pca.explained_variance_ratio_).round(2)):
            ax2.text(comp, t, t, fontsize=10)
        # Step12: Set the title
        title_str = (
            f"{task['ANALYSIS_NAME']}\n"
            "Multidimensional pattern in data via PCA (linear)\n"
        )
        plt.title(title_str, fontsize=10)
        # Step13: Save the figure
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_5_pca"  # noqa
        # Step14: Save as PNG
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # Step15: Optionally save as SVG
        if task["AS_SVG"]:
            # Make SCG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            plt.savefig(svg_save_path, bbox_inches="tight")
        # Step16: Display the figure
        plt.show()

    # --- Outlier Detection using Isolation Forest ---
    logging.info("Outlier Detection using Isolation Forest.")
    # If DATA_OUTLIER
    if task["DATA_OUTLIER"]:
        # Step1: Instantiate Isolation Forest
        iForest = IsolationForest(
            n_estimators=10000,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=-2,  # Use all available processors
            random_state=None,
            verbose=0,
            warm_start=False,
        )
        # Step2: Fit data and predict outliers
        outlier = iForest.fit_predict(x)
        # Step3: Create a DataFrame to store outlier labels
        outlier_df = pd.DataFrame(data=outlier, columns=["is_outlier"])
        # Step4: Compute outlier scores
        outlier_score = iForest.decision_function(x)
        # Step5: Plot outlier scores
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=outlier_score, bins=30, kde=True, color="#777777", ax=ax)
        # Step6: Format plot appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Isolation Forest outlier score")
        ax.set_ylabel("Count")
        # Step7: Calculate the percentage of potential outliers
        outlier_percentage = np.sum(outlier == -1) / len(outlier) * 100
        title_str = (
            f"{task['ANALYSIS_NAME']}\n"
            f"Outlier in data via Isolation Forest: {outlier_percentage:.1f}% potential outliers"  # noqa
        )
        plt.title(title_str, fontsize=10)
        # Step8: Save outputs
        # Create save path
        save_base_path = f"{task['store_path']}/{task['ANALYSIS_NAME']}_{task['y_name']}_eda_6_iForest"  # noqa
        # Step9: Save outlier data to an Excel file
        # Make excel save path
        excel_save_path = f"{save_base_path}.xlsx"
        # Save excel
        outlier_df.to_excel(excel_save_path, index=False)
        # Step10: Save the histogram plot as PNG
        # Make PNG save path
        png_save_path = f"{save_base_path}.png"
        # Save as PNG
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        # Step11: Optionally save as SVG
        if task["AS_SVG"]:
            # Make SVG save path
            svg_save_path = f"{save_base_path}.svg"
            # Save as SVG
            plt.savefig(svg_save_path, bbox_inches="tight")
        # Step12: Display the plot
        plt.show()


def main() -> None:
    """
    Main function of exploratory data analysis.

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
    OSError: If copy iml_1_eda script to store failed.
    FileNotFoundError: If load G, X, or Y failed.
    OSError: If save g, x, or y to store failed.
    OSError: If copy log file to store failed.
    """

    ####################################################################################
    # Script Configuration
    ####################################################################################

    # --- Specify task ---

    # Specify max number of samples. int (default: 10000)
    MAX_SAMPLES = 10000
    # Do 1D data distribution violin plot? bool (default: True)
    DATA_DISTRIBUTION_1D = True
    # Do 2D data distribution pair plot? bool (default: True)
    DATA_DISTRIBUTION_2D = True
    # Do pairwise correlations heatmap plot? bool (default: True)
    DATA_JOINT_INFORMATION_LINEAR = True
    # Do data pairwise prediction heatmap? (default: True)
    DATA_JOINT_INFORMATION_NON_LINEAR = True
    # Do multidimensional pattern heatmap with PCA? bool (default: True)
    DATA_MULTIDIM_PATTERN = True
    # Use Isolation Forest to detect outliers? bool (default: True)
    DATA_OUTLIER = True
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    N_JOBS = -2
    # Number of folds in CV. int (default: 5)
    N_CV_FOLDS = 5
    # Number of predictions in 5 fold CV. int (default: 1000)
    # Be aware of hardcoded min. 2 and max. 5 repetitions.
    N_PRED_CV = 1000
    # Device to run computations. str (default: "cuda", "cpu")
    DEVICE = "cuda"
    # Store prefix (where results go). str
    STORE_PREFIX = "iml_1_eda_"
    # Save plots additionally AS_SVG? bool (default: False)
    AS_SVG = False

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
    # Specify predictor names. list of str
    X_NAMES = [
        "nitrogen_nitrates",
        "nitrites_ammonia",
        "phosphate",
        "pH",
        "oxygen",
        "chloride",
        "compound_8",
        "season",
        "river_size",
        "fluid_velocity",
    ]
    # Specify target name(s). list of str or []
    Y_NAMES = [
        "concentration_a1",
        "concentration_a2",
    ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []

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
    # # Specify predictor names. list of str
    # X_NAMES = [
    #     "age",
    #     "bmi",
    #     "bp",
    #     "s1_tc",
    #     "s2_ldl",
    #     "s3_hdl",
    #     "s4_tch",
    #     "s5_ltg",
    #     "s6_glu",
    #     "gender",
    # ]
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "progression",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

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
    # # Specify predictor names. list of str
    # X_NAMES = [
    #     "age",
    #     "na_to_k",
    #     "gender_fm",
    #     "cholesterol_nh",
    #     "bp_lnh",
    # ]
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "drug",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

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
    # # Specify predictor names. list of str
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
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "attrition",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

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
    # # Specify predictor names. list of str
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
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "median_house_value",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Radon data - regression
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
    # # Specify predictor names. list of str
    # X_NAMES = [
    #     "uppm",
    #     "basement",
    #     "floor",
    #     "region",
    #     "room",
    #     "zip",
    # ]
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "log_radon",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

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
    # # Specify target name(s). list of str or []
    # Y_NAMES = [
    #     "maker",
    # ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

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
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # Log message format
        force=True,
    )
    # Create a console handler for output to the terminal
    console_handler = logging.StreamHandler()
    # Set the console log level
    console_handler.setLevel(logging.INFO)
    # Define the console log format
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Set the console log format
    console_handler.setFormatter(console_formatter)
    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    logging.info(
        f"Interpretable Machine Learning - Exploratory Data Analysis (EDA) of {ANALYSIS_NAME} started."  # noqa
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
        # Copy iml_1_eda script to store path
        shutil.copy("iml_1_eda.py", f"{store_path}/iml_1_eda.py")
    except OSError as e:
        # Raise error
        raise e

    # --- Task dictionary ---
    task = {
        "MAX_SAMPLES": MAX_SAMPLES,
        "N_JOBS": N_JOBS,
        "N_CV_FOLDS": N_CV_FOLDS,
        "DATA_DISTRIBUTION_1D": DATA_DISTRIBUTION_1D,
        "DATA_DISTRIBUTION_2D": DATA_DISTRIBUTION_2D,
        "DATA_JOINT_INFORMATION_LINEAR": DATA_JOINT_INFORMATION_LINEAR,
        "DATA_JOINT_INFORMATION_NON_LINEAR": DATA_JOINT_INFORMATION_NON_LINEAR,
        "DATA_MULTIDIM_PATTERN": DATA_MULTIDIM_PATTERN,
        "N_PRED_CV": N_PRED_CV,
        "DATA_OUTLIER": DATA_OUTLIER,
        "DEVICE": DEVICE,
        "STORE_PREFIX": STORE_PREFIX,
        "AS_SVG": AS_SVG,
        "ANALYSIS_NAME": ANALYSIS_NAME,
        "PATH_TO_DATA": PATH_TO_DATA,
        "SHEET_NAME": SHEET_NAME,
        "OBJECTIVE": OBJECTIVE,
        "G_NAME": G_NAME,
        "X_NAMES": X_NAMES,
        "Y_NAMES": Y_NAMES,
        "SKIP_ROWS": SKIP_ROWS,
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
    # Reindex X to X_NAMES
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

    # --- Prepare data ---
    # Iterate over prediction targets (Y_NAMES)
    for i_y, y_name in enumerate(Y_NAMES):
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
        g = G.reindex(index=y.index).reset_index(drop=True)
        # Use y index for predictors and reset index
        x = X.reindex(index=y.index).reset_index(drop=True)
        # Reset index of target
        y = y.reset_index(drop=True)
        # Log warning if samples were dropped because of NaNs in target
        if y.shape[0] < Y.shape[0]:
            # Lof warning
            logging.warning(
                f"{Y.shape[0] - y.shape[0]} samples were dropped due to NaNs in {y_name}."  # noqa
            )

        # --- Limit number of samples ---
        # Subsample predictors
        x = x.sample(
            n=min(x.shape[0], task["MAX_SAMPLES"]),
            random_state=None,
            ignore_index=False,
        )
        # Slice group to fit subsampled predictors
        g = g.loc[x.index, :].reset_index(drop=True)
        # Slice targets to fit subsampled predictors
        y = y.loc[x.index, :].reset_index(drop=True)
        # Reset index of predictors
        x = x.reset_index(drop=True)

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

        # --- Run Exploratory Data Analysis (EDA) ---
        eda(task, g, x, y)

    # --- Save log file ---
    # Log success
    logging.info(f"Exploratory Data Analysis (EDA) of {ANALYSIS_NAME} finished.")
    try:
        # Copy log file to results directory
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
