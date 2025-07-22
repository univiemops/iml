# -*- coding: utf-8 -*-
"""
Interpretable Machine-Learning 3 - Plotting (PLT)
v400
@author: david.steyrl@univie.ac.at
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import shutil
import subprocess
from scipy.stats import t
from shap import Explanation
from shap.plots import beeswarm
from shap.plots import scatter
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from typing import Callable

# Set plot log level to Warning (Info not shown anymore)
plt.set_loglevel("WARNING")


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


def corrected_std(differences: np.ndarray, n_tst_over_n_trn: float = 0.25) -> float:
    """
    Corrects standard deviation using Nadeau and Bengio's approach.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.

    Raises
    ------
    None
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)
    # Return corrected standard deviation
    return corrected_std


def corrected_ttest(differences: np.ndarray, n_tst_over_n_trn: float = 0.25) -> float:
    """
    Computes right-tailed paired t-test with corrected variance.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.

    Raises
    ------
    None
    """
    # Get mean of differences
    mean = np.nanmean(differences)
    # Get corrected standard deviation, make sure std is not exactly zero
    std = max(1e-6, corrected_std(differences, n_tst_over_n_trn))
    # Compute t statistics
    t_stat = mean / std
    # Compute p value for one-tailed t-test
    p_val = t.sf(t_stat, df=len(differences) - 1)
    # Return t statistics and p value
    return t_stat, p_val


def plot_parameter_distributions(task: dict, results: dict, store_path: str) -> None:
    """
    Print model parameter distributions in histogram.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Check if best_params in in results
    if "best_params" in results.keys():
        # Get params
        params = pd.DataFrame(results["best_params"])
    else:
        # Log warning
        logging.warning("No best_params found in results. Skip plot.")
        # Return
        return

    # --- Make plot ---
    # Iterate over columns of params dataframe
    for idx, (name, data) in enumerate(params.items()):
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(
            data=data.astype("float"),
            bins=30,
            kde=True,
            color="#777777",
            log_scale=True if name.endswith("reg_lambda") else False,
            ax=ax,
        )
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel("Count")
        # Set x range
        if name.endswith("colsample_bytree"):
            ax.set_xlim([0, 1])
        elif name.endswith("extra_trees"):
            ax.set_xlim([-0.1, 1.1])
        elif name.endswith("reg_lambda"):
            ax.set_xlim([0.1, 100])
        # Set title
        ax.set_title(
            f"{task['ANALYSIS_NAME']}\nParameter distribution of predicting {task['y_name']}",  # noqa
            fontsize=10,
        )

        # --- Save figure ---
        # Make save path
        save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_0_{idx}_parameter_{name}"[  # noqa
            :150
        ]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # If save as svg
        if task["AS_SVG"]:
            # Save figure
            plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        # Show figure
        plt.show()


def plot_regression_scatter(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [pd.Series(k["y_true"]).map(RULE) for k in results["scores"]]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k["x_tst"][DIVIDE_DATA_BY].map(RULE) for k in results["scores"]]
    else:
        # Masks are all True to pass all data
        masks = [
            pd.Series(k["y_true"]).map(lambda item: True) for k in results["scores"]
        ]

    # --- Prepare results ---
    # True values
    true_values_per_fold = [k["y_true"][m] for k, m in zip(results["scores"], masks)]
    # Predicted values
    pred_values_per_fold = [k["y_pred"][m] for k, m in zip(results["scores"], masks)]
    # True values
    true_values = np.concatenate(true_values_per_fold)
    # Predicted values
    pred_values = np.concatenate(pred_values_per_fold)
    # True values shuffle
    true_values_per_fold_sh = [
        k["y_true"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Predicted values shuffle
    pred_values_per_fold_sh = [
        k["y_pred"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Compute MAE
    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Extract MAE shuffle
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Extract R²
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    # Extract R² shuffle
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]

    # --- Make plot ---
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(
        pred_values,
        true_values,
        zorder=2,
        alpha=0.1,
        color="#444444",
    )
    # Add optimal fit line
    ax.plot(
        [-10000, 10000],
        [-10000, 10000],
        color="#999999",
        zorder=3,
        linewidth=2,
        alpha=0.3,
    )
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Remove ticks
    ax.tick_params(
        axis="both",
        which="major",
        reset=True,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    # Add grid
    ax.grid(visible=True, which="major", axis="both")
    # Modify grid
    ax.tick_params(grid_linestyle=":", grid_alpha=0.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    # Set y-axis limits
    ax.set_ylim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    # Set title
    ax.set_title(
        f"{task['ANALYSIS_NAME']}\nPredicting {task['y_name']}",
        fontsize=10,
    )
    # Set xlabel
    ax.set_xlabel(f"Predicted {task['y_name']}", fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel(f"True {task['y_name']}", fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # --- Add MAE ---
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh) - np.array(mae))
    # Add original outcome MAE results to plot
    ax.text(
        0.3,
        0.09,
        f"Original data: MAE mean{r'$\pm$'}std {np.nanmean(mae):.2f}{r'$\pm$'}{np.std(mae):.2f} | med {np.nanmedian(mae):.2f}",  # noqa
        transform=ax.transAxes,
        fontsize=10,
    )
    # Add shuffled outcome MAE results to plot
    ax.text(
        0.3,
        0.055,
        f"Shuffled data: MAE mean{r'$\pm$'}std {np.nanmean(mae_sh):.2f}{r'$\pm$'}{np.std(mae_sh):.2f} | med {np.nanmedian(mae_sh):.2f}",  # noqa
        transform=ax.transAxes,
        fontsize=10,
    )
    # If pval_mae <= 0.001
    if pval_mae <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = f"p={pval_mae:.3f}"
    # Add p value to the plot
    ax.text(
        0.3,
        0.02,
        f"Original vs. shuffled: {pval_string}",
        transform=ax.transAxes,
        fontsize=10,
    )

    # --- Add R² ---
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2) - np.array(r2_sh))
    # Add original outcome R² results to plot
    ax.text(
        0.02,
        0.96,
        f"Original data: R² mean{r'$\pm$'}std {np.nanmean(r2):.3f}{r'$\pm$'}{np.std(r2):.3f} | med {np.nanmedian(r2):.3f}",  # noqa
        transform=ax.transAxes,
        fontsize=10,
    )
    # Add shuffled outcome R² results to plot
    ax.text(
        0.02,
        0.925,
        f"Shuffled data: R² mean{r'$\pm$'}std {np.nanmean(r2_sh):.3f}{r'$\pm$'}{np.std(r2_sh):.3f} | med {np.nanmedian(r2_sh):.3f}",  # noqa
        transform=ax.transAxes,
        fontsize=10,
    )
    # If pval_r2 <= 0.001
    if pval_r2 <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = "p={:.3f}".format(pval_r2)
    # Add p value to the plot
    ax.text(
        0.02,
        0.89,
        f"Original vs. shuffled: {pval_string}",
        transform=ax.transAxes,
        fontsize=10,
    )

    # --- Save figure ---
    # Make save path
    save_path = (
        f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_1_0_predictions"[:150]
    )  # noqa
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # If save as svg
    if task["AS_SVG"]:
        # Save figure
        plt.savefig(f"{save_path}.svg", bbox_inches="tight")
    # Show figure
    plt.show()


def plot_regression_violin(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [pd.Series(k["y_true"]).map(RULE) for k in results["scores"]]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k["x_tst"][DIVIDE_DATA_BY].map(RULE) for k in results["scores"]]
    else:
        # Masks are all True to pass all data
        masks = [
            pd.Series(k["y_true"]).map(lambda item: True) for k in results["scores"]
        ]

    # --- Prepare results ---
    # True values
    true_values_per_fold = [k["y_true"][m] for k, m in zip(results["scores"], masks)]
    # Predicted values
    pred_values_per_fold = [k["y_pred"][m] for k, m in zip(results["scores"], masks)]
    # True values shuffle
    true_values_per_fold_sh = [
        k["y_true"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Predicted values shuffle
    pred_values_per_fold_sh = [
        k["y_pred"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Compute MAE
    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Extract MAE shuffle
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Extract R²
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    # Extract R² shuffle
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae)),
            "R²": pd.Series(np.array(r2)),
            "Data": pd.Series(["original" for _ in mae]),
            "Dummy": pd.Series(np.ones(np.array(mae).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae_sh)),
            "R²": pd.Series(np.array(r2_sh)),
            "Data": pd.Series(["shuffled" for _ in mae_sh]),
            "Dummy": pd.Series(np.ones(np.array(mae_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Mean Absolute Error", "R²"]

    # --- Make plot ---
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics) * 1 + 1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(
            x=metric,
            y="Dummy",
            hue="Data",
            data=all_scores_df,
            bw_method="scott",
            bw_adjust=0.5,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(
            axis="y",
            color="#bbbbbb",
            linestyle="dotted",
            alpha=0.3,
        )
    # Make title string
    title_str = f"{task['ANALYSIS_NAME']}\nPredicting {task['y_name']}"
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # --- Save figure ---
    # Make save path
    save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_1_1_predictions_distribution"[  # noqa
        :150
    ]
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # If save as svg
    if task["AS_SVG"]:
        # Save figure
        plt.savefig(f"{save_path}.svg", bbox_inches="tight")
    # Show plot
    plt.show()


def plot_classification_confusion(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Model fit as confusion matrix plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [pd.Series(k["y_true"]).map(RULE) for k in results["scores"]]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k["x_tst"][DIVIDE_DATA_BY].map(RULE) for k in results["scores"]]
    else:
        # Masks are all True to pass all data
        masks = [
            pd.Series(k["y_true"]).map(lambda item: True) for k in results["scores"]
        ]

    # --- Prepare results ---
    # True values
    true_values_per_fold = [k["y_true"][m] for k, m in zip(results["scores"], masks)]
    # Predicted values
    pred_values_per_fold = [k["y_pred"][m] for k, m in zip(results["scores"], masks)]
    # True values
    true_values = np.concatenate(true_values_per_fold)
    # Predicted values
    pred_values = np.concatenate(pred_values_per_fold)
    # True values shuffle
    true_values_per_fold_sh = [
        k["y_true"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Predicted values shuffle
    pred_values_per_fold_sh = [
        k["y_pred"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Accuracy
    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Schuffle accuracy
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Get classes
    class_labels = np.unique(true_values)

    # --- Get count confusion matrix ---
    # Loop over single results
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        if "con_mat_count" not in locals():
            # Compute confusion matrix
            con_mat_count = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize=None,
            )
        else:
            # Add confusion matrix
            con_mat_count = np.add(
                con_mat_count,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize=None,
                ),
            )

    # --- Get normalized confusion matrix ---
    # Loop over single results
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        if "con_mat" not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize="true",
            )
        else:
            # Add confusion matrix
            con_mat = np.add(
                con_mat,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize="true",
                ),
            )
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values_per_fold)

    # --- Plot confusion matrix ---
    # Create figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(con_mat.shape[0] * 0.5 + 4, con_mat.shape[0] * 0.5 + 3.5),
    )
    # Use tight layout
    plt.tight_layout()
    # Plot count confusion matrix
    sns.heatmap(
        con_mat_count,
        vmin=None,
        vmax=None,
        cmap="Greys",
        center=None,
        robust=True,
        annot=True,
        fmt="",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[0],
    )
    # Add title to plot
    ax[0].set_title("num. predictions", fontsize=10)
    # Add x label to plot
    ax[0].set_xlabel(f"Predicted {task['y_name']}", fontsize=10)
    # Add y label to plot
    ax[0].set_ylabel(f"True {task['y_name']}", fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[0].tick_params(axis="y", labelsize=10, labelrotation=0)
    # Plot normalized confusion matrix
    sns.heatmap(
        con_mat_norm * 100,
        vmin=None,
        vmax=None,
        cmap="Greys",
        center=None,
        robust=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[1],
    )
    # Add title to plot
    ax[1].set_title(f"norm. to True {task['y_name']}", fontsize=10)
    # Add x label to plot
    ax[1].set_xlabel(f"Predicted {task['y_name']}", fontsize=10)
    # Add y label to plot
    ax[1].set_ylabel(f"True {task['y_name']}", fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[1].tick_params(axis="y", labelsize=10, labelrotation=0)
    # Calculate p-value of accuracy and shuffle accuracy
    tstat_acc, pval_acc = corrected_ttest(np.array(acc) - np.array(acc_sh))
    # If pval_acc <= 0.001
    if pval_acc <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = "p={:.3f}".format(pval_acc)
    # Make title string
    title_str = (
        f"{task['ANALYSIS_NAME']}"
        + f"\nPredicting {task['y_name']}"
        + f"\nOriginal data acc: mean{r'$\pm$'}std {np.nanmean(acc):.3f}{r'$\pm$'}{np.std(acc):.3f} | med {np.nanmedian(acc):.3f}"  # noqa
        + f"\nShuffled data acc: mean{r'$\pm$'}std {np.nanmean(acc_sh):.3f}{r'$\pm$'}{np.std(acc_sh):.3f} | med {np.nanmedian(acc_sh):.3f}"  # noqa
        + f"\nOriginal vs. shuffled: {pval_string}"
    )
    # Set title
    plt.suptitle(title_str, fontsize=10, y=0.95)

    # --- Save figure ---
    # Make save path
    save_path = (
        f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_1_0_predictions"[:150]
    )
    # Save figure
    plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
    # If save as svg
    if task["AS_SVG"]:
        # Save figure
        plt.savefig(save_path + ".svg", bbox_inches="tight")
    # Show figure
    plt.show()


def plot_classification_violin(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.


    Returns
    -------
    None

    Raises
    ------
    None
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [pd.Series(k["y_true"]).map(RULE) for k in results["scores"]]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k["x_tst"][DIVIDE_DATA_BY].map(RULE) for k in results["scores"]]
    else:
        # Masks are all True to pass all data
        masks = [
            pd.Series(k["y_true"]).map(lambda item: True) for k in results["scores"]
        ]

    # --- Prepare results ---
    # True values
    true_values_per_fold = [k["y_true"][m] for k, m in zip(results["scores"], masks)]
    # Predicted values
    pred_values_per_fold = [k["y_pred"][m] for k, m in zip(results["scores"], masks)]
    # True values shuffle
    true_values_per_fold_sh = [
        k["y_true"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Predicted values shuffle
    pred_values_per_fold_sh = [
        k["y_pred"][m] for k, m in zip(results["scores_sh"], masks)
    ]
    # Accuracy
    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Schuffle accuracy
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc)),
            "Data": pd.Series(["original" for _ in acc]),
            "Dummy": pd.Series(np.ones(np.array(acc).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc_sh)),
            "Data": pd.Series(["shuffled" for _ in acc_sh]),
            "Dummy": pd.Series(np.ones(np.array(acc_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Accuracy"]

    # --- Make plot ---
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics) * 1 + 1))
    # Make color palette
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(
            x=metric,
            y="Dummy",
            hue="Data",
            data=all_scores_df,
            bw_method="scott",
            bw_adjust=0.5,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
    # Make title string
    title_str = f"{task['ANALYSIS_NAME']}\nPredicting {task['y_name']}"
    # set title
    plt.title(title_str, fontsize=10)

    # --- Save figure ---
    # Make save path
    save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_1_1_predictions_distribution"[  # noqa
        :150
    ]
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # If save as svg
    if task["AS_SVG"]:
        # Save figure
        plt.savefig(f"{save_path}.svg", bbox_inches="tight")
    # Show plot
    plt.show()


def get_avg_shap_values(
    task: dict,
    explanations: list,
    c_class: int = -1,
    DIVIDE_DATA_BY: str = "",
    RULE: Callable = lambda item: True,
) -> tuple:
    """
    Get average SHAP values (global effects).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explanations : list of shap explanation objects
        SHAP explanation holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    shap_values : list
        SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [k.labels[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k.data[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    else:
        # Masks are all True to pass all data
        masks = [k.labels.map(lambda item: True).values.squeeze() for k in explanations]

    # --- Get shap values ---
    # If regression
    if task["OBJECTIVE"] == "regression":

        # --- Get Shap values ---
        # If no interactions
        if len(explanations[0].shape) == 2:
            # Get average SHAP values
            shap_values = [
                np.nanmean(np.abs(k.values[m]), axis=0)
                for k, m in zip(explanations, masks)
            ]
        # If interactions
        elif len(explanations[0].shape) == 3:
            # Get average SHAP values
            shap_values = [
                np.nanmean(np.abs(np.sum(k.values[m], axis=2)), axis=0)
                for k, m in zip(explanations, masks)
            ]
        # Average base value
        base = np.nanmean(
            np.hstack([k.base_values[m] for k, m in zip(explanations, masks)])
        )
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # If no interactions
        if len(explanations[0].shape) == 3:
            # SHAP values
            shap_values = [
                np.nanmean(np.abs(k.values[m, :, c_class]), axis=0)
                for k, m in zip(explanations, masks)
            ]
            base = np.nanmean(
                np.hstack(
                    [k[m, :, c_class].base_values for k, m in zip(explanations, masks)]
                )
            )
        # If interactions
        elif len(explanations[0].shape) == 4:
            # Get SHAP values
            shap_values = [
                np.nanmean(np.abs(np.sum(k.values[m, :, :, c_class], axis=2)), axis=0)
                for k, m in zip(explanations, masks)
            ]
            # Base value
            base = np.nanmean(
                np.hstack(
                    [
                        k[m, :, :, c_class].base_values
                        for k, m in zip(explanations, masks)
                    ]
                )
            )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")
    # Make SHAP values dataframe
    shap_values_df = pd.DataFrame(shap_values, columns=explanations[0].feature_names)

    # --- Return shap values ---
    return shap_values_df, base


def plot_avg_shap_values(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot average SHAP values (global effects).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values ---
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values_df, base = get_avg_shap_values(
            task,
            results["explanations"],
            c_class,
            DIVIDE_DATA_BY,
            RULE,
        )
        # Get current shuffle shap values
        shap_values_sh_df, _ = get_avg_shap_values(
            task,
            results["explanations_sh"],
            c_class,
            DIVIDE_DATA_BY,
            RULE,
        )
        # --- Process SHAP values ---
        # Mean shap values
        shap_values_se_mean = shap_values_df.mean(axis=0)
        # Sort from high to low
        shap_values_se_mean_sort = shap_values_se_mean.sort_values(ascending=True)

        # --- Additional info ---
        # x names lengths
        x_names_max_len = max([len(i) for i in shap_values_df.columns.to_list()])
        # x names count
        x_names_count = len(shap_values_df.columns.to_list())
        # Make title string
        title_str = (
            f"{task['ANALYSIS_NAME']}"
            + f"\nmean(|SHAP values|): average effect on {task['y_name']}"
            + f"\nmean(|SHAP values|): average change from expected value of {np.round(base, decimals=2)}"  # noqa
        )
        # Add class if classification
        if task["OBJECTIVE"] == "classification":
            # Make title string
            title_str = f"{title_str} (log odds)\n for class: {c_class}"
        # Get number of lines of title string
        title_lines_count = title_str.count("\n") + 1

        # --- Plot ---
        # Make horizontal bar plot
        shap_values_se_mean_sort.plot(
            kind="barh",
            figsize=(
                x_names_max_len * 0.1 + 8,
                x_names_count * 0.4 + title_lines_count * 0.4 + 0.5,
            ),
            color="#777777",
            fontsize=10,
        )
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("mean(|SHAP values|)", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        # Set title
        ax.set_title(title_str, fontsize=10)

        # --- Compute SHAP values p values ---
        # Init p value list
        pval = []
        # Iterate over predictors
        for pred_name, pred_data in shap_values_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy() - shap_values_sh_df[pred_name].to_numpy()
            )
            # Add to pval list
            pval.append(np.around(c_pval, decimals=3))
        # Make pval series
        pval_se = pd.Series(data=pval, index=shap_values_df.columns.to_list())
        # Multiple comparison correction
        if task["MCC"]:
            # Multiply p value by number of tests
            pval_se = pval_se * x_names_count
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)

        # --- Add SHAP values and p values as text ---
        # Loop over values
        for i, (c_pred, c_val) in enumerate(shap_values_se_mean_sort.items()):
            # If pval_se[c_pred] <= 0.001
            if pval_se[c_pred] <= 0.001:
                # Make pval string
                pval_string = "p\u22640.001"
            else:
                # Make pval string
                pval_string = "p={:.3f}".format(pval_se[c_pred])
            # Make text string
            txt_str = ("{:.2f}" + " | " + pval_string).format(c_val)
            # Add values to plot
            ax.text(
                c_val + shap_values_se_mean.max() * 0.01,
                i,
                txt_str,
                color="k",
                va="center",
                fontsize=10,
            )
        # Get x limits
        x_left, x_right = plt.xlim()
        # Set x limits
        plt.xlim(x_left, x_right + x_right * 0.15)

        # --- Save plot ---
        # Make save path
        save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_2_0_{c_class}_avg_shap_values"[  # noqa
            :150
        ]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # If save as svg
        if task["AS_SVG"]:
            # Save figure
            plt.savefig("{save_path}.svg", bbox_inches="tight")
        # Show figure
        plt.show()


def plot_avg_shap_values_distributions(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot average SHAP values distributions (global effects distribution).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations is in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values distribution ---
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values_df, base = get_avg_shap_values(
            task,
            results["explanations"],
            c_class,
        )
        # Get current shuffle shap values
        shap_values_sh_df, _ = get_avg_shap_values(
            task,
            results["explanations_sh"],
            c_class,
        )

        # --- Process SHAP values ---
        # Sorting index by mean value of columns
        i_srt = shap_values_df.mean().sort_values(ascending=False).index
        # Sort SHAP values dataframe
        shap_values_df_sort = shap_values_df.reindex(i_srt, axis=1)
        # Sort shuffle SHAP values dataframe
        shap_values_sh_df_sort = shap_values_sh_df.reindex(i_srt, axis=1)
        # Add data origin to SHAP values dataframe
        shap_values_df_sort["Data"] = pd.DataFrame(
            ["original" for _ in range(shap_values_df_sort.shape[0])],
            columns=["Data"],
        )
        # Add data origin to shuffle SHAP values dataframe
        shap_values_sh_df_sort["Data"] = pd.DataFrame(
            ["shuffled" for _ in range(shap_values_sh_df_sort.shape[0])],
            columns=["Data"],
        )
        # Get value name
        value_name = "mean(|SHAP values|)"
        # Melt SHAP values dataframe
        shap_values_df_sort_melt = shap_values_df_sort.melt(
            id_vars=["Data"],
            var_name="predictors",
            value_name=value_name,
        )
        # Melt shuffle SHAP values dataframe
        shap_values_sh_df_sort_melt = shap_values_sh_df_sort.melt(
            id_vars=["Data"],
            var_name="predictors",
            value_name=value_name,
        )
        # Concatenate importances dataframes
        shap_values_df_sort_melt_all = pd.concat(
            [shap_values_df_sort_melt, shap_values_sh_df_sort_melt],
            axis=0,
        )

        # --- Additional info ---
        # x names lengths
        x_names_max_len = max([len(i) for i in shap_values_df.columns.to_list()])
        # x names count
        x_names_count = len(shap_values_df.columns.to_list())
        # Make title string
        title_str = (
            f"{task['ANALYSIS_NAME']}"
            + f"\nmean(|SHAP values|): average effect on {task['y_name']}"
            + f"\nmean(|SHAP values|): average change from expected value of {np.round(base, decimals=2)}"  # noqa
        )
        # Add class if multiclass
        if task["OBJECTIVE"] == "classification":
            # Make title string
            title_str = f"{title_str} (log odds)\n class: {c_class}"
        # Get number of lines of title string
        title_lines_count = title_str.count("\n") + 1

        # --- Plot ---
        # Make figure
        fig, ax = plt.subplots(
            figsize=(
                x_names_max_len * 0.1 + 8,
                x_names_count * 0.4 + title_lines_count * 0.4 + 0.5,
            ),
        )
        # Make color palette
        mypal = {"original": "#777777", "shuffled": "#eeeeee"}
        # Plot data
        sns.violinplot(
            x=value_name,
            y="predictors",
            hue="Data",
            data=shap_values_df_sort_melt_all,
            bw_method="scott",
            bw_adjust=0.5,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=0.5,
            saturation=1,
            ax=ax,
            palette=mypal,
        )
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("mean(|SHAP values|)", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel("", fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        # Set legend position
        plt.legend(loc="lower right")
        # Add title
        ax.set_title(title_str, fontsize=10)

        # --- Save plots and results ---
        # Make save path
        save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_2_1_{c_class}_avg_shap_values_distributions"[  # noqa
            :150
        ]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # If save as svg
        if task["AS_SVG"]:
            # Save figure
            plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        # Show figure
        plt.show()


def get_single_shap_values(
    task: dict,
    explanations: list,
    c_class: int = -1,
    DIVIDE_DATA_BY: str = "",
    RULE: Callable = lambda item: True,
) -> tuple:
    """
    Get single SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explanations : list of shap explanation objects
        SHAP explanation holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    shap_explanations : shap explanation object
        explanation object with SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [k.labels[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k.data[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    else:
        # Masks are all True to pass all data
        masks = [k.labels.map(lambda item: True).values.squeeze() for k in explanations]

    # --- Get shap values ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Explainer object
        shap_explanations = Explanation(
            np.vstack([k.values[m] for k, m in zip(explanations, masks)]),
            base_values=np.hstack(
                [k.base_values[m] for k, m in zip(explanations, masks)]
            ),
            data=np.vstack([k.data[m] for k, m in zip(explanations, masks)]),
            display_data=None,
            instance_names=None,
            feature_names=explanations[0].feature_names,
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
        # Base value
        base = np.nanmean(
            np.hstack([k.base_values[m] for k, m in zip(explanations, masks)])
        )
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # If no interactions
        if len(explanations[0].shape) == 3:
            # Explainer object
            shap_explanations = Explanation(
                np.vstack(
                    [k[m, :, c_class].values for k, m in zip(explanations, masks)]
                ),
                base_values=np.hstack(
                    [k[m, :, c_class].base_values for k, m in zip(explanations, masks)]
                ),
                data=np.vstack(
                    [k[m, :, c_class].data for k, m in zip(explanations, masks)]
                ),
                display_data=None,
                instance_names=None,
                feature_names=explanations[0].feature_names,
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
            # Base value
            base = np.nanmean(
                np.hstack(
                    [k[m, :, c_class].base_values for k, m in zip(explanations, masks)]
                )
            )
        # If no interactions
        elif len(explanations[0].shape) == 4:
            # Explainer object
            shap_explanations = Explanation(
                np.vstack(
                    [k[m, :, :, c_class].values for k, m in zip(explanations, masks)]
                ),
                base_values=np.hstack(
                    [
                        k[m, :, :, c_class].base_values
                        for k, m in zip(explanations, masks)
                    ]
                ),
                data=np.vstack(
                    [k[m, :, :, c_class].data for k, m in zip(explanations, masks)]
                ),
                display_data=None,
                instance_names=None,
                feature_names=explanations[0].feature_names,
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
            # Base value
            base = np.nanmean(
                np.hstack(
                    [
                        k[m, :, :, c_class].base_values
                        for k, m in zip(explanations, masks)
                    ]
                )
            )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Return shap values ---
    return shap_explanations, base


def plot_single_shap_values(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot single SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations is in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values ---
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        single_shap_values, base = get_single_shap_values(
            task,
            results["explanations"],
            c_class,
            DIVIDE_DATA_BY,
            RULE,
        )
        # If no interactions
        if len(single_shap_values.shape) == 2:
            # Take single shap values as is
            explanations = single_shap_values
        # If interactions
        elif len(single_shap_values.shape) == 3:
            # Sum over interaction to get total values
            explanations = single_shap_values.sum(axis=2)
            # Add base values
            explanations.base_values = single_shap_values.base_values
            # Add data
            explanations.data = single_shap_values.data

        # --- Additional info ---
        # x names lengths
        x_names_max_len = max([len(i) for i in single_shap_values.feature_names])
        # x names count
        x_names_count = len(single_shap_values.feature_names)
        # Make title string
        title_str = (
            f"{task['ANALYSIS_NAME']}"
            + f"\nSHAP values: single effects on {task['y_name']}"
            + f"\nSHAP values: change from expected value of {np.round(base, decimals=2)}"  # noqa
        )
        # Add class if multiclass
        if task["OBJECTIVE"] == "classification":
            # Make title string
            title_str = f"{title_str} (log odds)\n class: {c_class}"
        # Get number of lines of title string
        title_lines_count = title_str.count("\n") + 1

        # --- Plot SHAP values beeswarm ----
        beeswarm(
            explanations,
            max_display=len(single_shap_values.feature_names),
            order=Explanation.abs.mean(0),
            clustering=None,
            cluster_threshold=0.5,
            color=None,
            axis_color="#333333",
            alpha=0.66,
            show=False,
            log_scale=False,
            color_bar=True,
            plot_size=(
                x_names_max_len * 0.1 + 8,
                x_names_count * 0.4 + title_lines_count * 0.4 + 0.5,
            ),
            color_bar_label="Predictor value",
        )
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("SHAP values", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel("Predictor value", fontsize=10)

        # --- Save plot ---
        # Make save path
        save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_2_2_{c_class}_shap_values"[  # noqa
            :150
        ]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # If save as svg
        if task["AS_SVG"]:
            # Save figure
            plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        # Show figure
        plt.show()


def plot_single_shap_values_dependences(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot single SHAP values dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations is in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values ---
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        single_shap_values, base = get_single_shap_values(
            task,
            results["explanations"],
            c_class,
            DIVIDE_DATA_BY,
            RULE,
        )
        # If no interactions
        if len(single_shap_values.shape) == 2:
            # Take single shap values as is
            explanations = single_shap_values
        # If interactions
        elif len(single_shap_values.shape) == 3:
            # Sum over interaction to get total values
            explanations = single_shap_values.sum(axis=2)
            # Add base values
            explanations.base_values = single_shap_values.base_values
            # Add data
            explanations.data = single_shap_values.data

        # --- Print shap values dependencies ---
        # Loop over predictors
        for idx, c_pred in enumerate(explanations.feature_names):
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Make title string
            title_str = (
                f"{task['ANALYSIS_NAME']}"
                + f"\nSHAP values: single effects on {task['y_name']}"
                + f"\nSHAP values: change from expected value of {np.round(base, decimals=2)}"  # noqa
            )
            # Add class if multiclass
            if task["OBJECTIVE"] == "classification":
                # Make title string
                title_str = f"{title_str} (log odds)\n class: {c_class}"
            # Plot SHAP Scatter plot
            scatter(
                explanations[:, idx],
                color="#777777",
                hist=True,
                axis_color="#333333",
                dot_size=16,
                x_jitter="auto",
                alpha=0.5,
                title=title_str,
                xmin=None,
                xmax=None,
                ymin=None,
                ymax=None,
                overlay=None,
                ax=ax,
                show=False,
            )
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set title size
            ax.title.set_size(10)
            # Set x label size
            plt.xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Make y label
            y_label = f"Effects on {task['y_name']} (SHAP values)"
            # Set y label size
            plt.ylabel(y_label, fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)

            # --- Save plot ---
            # Make save path
            save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_3_{c_class}_{idx}_shap_values_dependency_{c_pred}"[  # noqa
                :150
            ]
            # Save figure
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
            # If save as svg
            if task["AS_SVG"]:
                # Save figure
                plt.savefig(f"{save_path}.svg", bbox_inches="tight")
            # Show figure
            plt.show()


def get_avg_shap_interaction_values(
    task: dict,
    explanations: list,
    c_class: int = -1,
    DIVIDE_DATA_BY: str = "",
    RULE: Callable = lambda item: True,
) -> tuple:
    """
    Get average SHAP interaction values (global interaction effects).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explanations : list of shap explanation objects
        SHAP explanation holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    shap_values : list
        SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """

    # --- Get masks ---
    # If divide variable is in prediction target
    if DIVIDE_DATA_BY == task["y_name"]:
        # Masks are based in laberls
        masks = [k.labels[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    # If divide variable is in predictors
    elif DIVIDE_DATA_BY in task["X_NAMES"]:
        # Masks are based on data
        masks = [k.data[DIVIDE_DATA_BY].map(RULE).values for k in explanations]
    else:
        # Masks are all True to pass all data
        masks = [k.labels.map(lambda item: True).values.squeeze() for k in explanations]

    # --- Get shap interaction values ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Get SHAP interaction values
        shap_values_inter = np.array(
            [
                np.nanmean(np.abs(k.values[m]), axis=0)
                for k, m in zip(explanations, masks)
            ]
        )
        # Base value
        base = np.nanmean(
            np.hstack([k.base_values[m] for k, m in zip(explanations, masks)])
        )
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Get SHAP interaction values
        shap_values_inter = np.array(
            [
                np.nanmean(np.abs(k[m, :, :, c_class].values), axis=0)
                for k, m in zip(explanations, masks)
            ]
        )
        # Base value
        base = np.nanmean(
            np.hstack(
                [k[m, :, :, c_class].base_values for k, m in zip(explanations, masks)]
            )
        )
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Return shap interaction values ---
    return shap_values_inter, base


def plot_average_shap_interaction_values(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot average SHAP interaction values (global interaction effects).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations is in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return
    # If regression
    if task["OBJECTIVE"] == "regression":
        # If no interactions
        if len(results["explanations"][0].shape) == 2:
            # Log warning
            logging.warning("No SHAP interaction values found. Skip plot.")
            # Return
            return
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # If no interactions
        if len(results["explanations"][0].shape) == 3:
            # Log warning
            logging.warning("No SHAP interaction values found. Skip plot.")
            # Return
            return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values ---
    for c_class in range(n_classes):
        # Get current shap values
        shap_values_df, base = get_avg_shap_values(
            task,
            results["explanations"],
            c_class,
        )

        # --- Process SHAP values ---
        # Mean shap values
        shap_values_se_mean = shap_values_df.mean(axis=0)
        # Sort from high to low
        shap_values_se_mean_sort = shap_values_se_mean.sort_values(ascending=False)

        # --- Get SHAP values interactions ---
        # SHAP values
        shap_values_inter, base_inter = get_avg_shap_interaction_values(
            task,
            results["explanations"],
            c_class,
        )
        # Make dataframe
        shap_values_inter_df = pd.DataFrame(
            np.nanmean(shap_values_inter, axis=0),
            index=shap_values_df.columns.to_list(),
            columns=shap_values_df.columns.to_list(),
        )
        # Reindex to sorted index
        shap_values_inter_sort_df = shap_values_inter_df.reindex(
            shap_values_se_mean_sort.index
        )
        # Reorder columns to sorted index
        shap_values_inter_sort_df = shap_values_inter_sort_df.loc[
            :, shap_values_se_mean_sort.index
        ]
        # SHAP values shuffle
        shap_values_inter_sh, base_inter_sh = get_avg_shap_interaction_values(
            task,
            results["explanations_sh"],
            c_class,
        )

        # --- Additional info ---
        # x names lengths
        x_names_max_len = max([len(i) for i in shap_values_df.columns.to_list()])
        # x names count
        x_names_count = len(shap_values_df.columns.to_list())
        # Make title string
        title_str = (
            f"{task['ANALYSIS_NAME']}"
            + f"\nmean(|SHAP values|): average effects on {task['y_name']}"
            + f"\nmean(|SHAP values|): average change from expected value of {np.round(base, decimals=2)}"  # noqa
        )
        # Add class if multiclass
        if task["OBJECTIVE"] == "classification":
            # Make title string
            title_str = f"{title_str} (log odds)\n class: {c_class}"
        # Get number of lines of title string
        title_lines_count = title_str.count("\n") + 1

        # --- Make labels with pvales ---
        # Init p values
        pval = np.zeros(
            (
                shap_values_inter.shape[1],
                shap_values_inter.shape[2],
            ),
        )
        # Iterate over shap_values
        for x, y in np.ndindex(
            (shap_values_inter.shape[1], shap_values_inter.shape[2])
        ):
            # Get current SHAP values
            c_value = shap_values_inter[:, x, y]
            # Get current SHAP values shuffle
            c_value_sh = shap_values_inter_sh[:, x, y]
            # Calculate p-value
            _, pval[x, y] = corrected_ttest(c_value - c_value_sh)
        # Multiple comparison correction
        if task["MCC"]:
            # Multiply p value by number of tests
            pval = pval * (x_names_count**2)
            # Set p values > 1 to 1
            pval = pval.clip(None, 1)
        # Initialize labels dataframe
        interaction_labels_df = pd.DataFrame(
            np.zeros(
                [shap_values_inter.shape[1], shap_values_inter.shape[2]],
            ),
            dtype="string",
        )
        # Iterate labels
        for x, y in np.ndindex(
            (shap_values_inter.shape[1], shap_values_inter.shape[2])
        ):
            # If pval[x, y] <= 0.001
            if pval[x, y] <= 0.001:
                # Make pval string
                pval_string = "p\u22640.001"
            else:
                # Make pval string
                pval_string = "p={:.3f}".format(pval[x, y])
            # Make label
            interaction_labels_df.iloc[x, y] = ("{:.2f}" + "\n" + pval_string).format(
                shap_values_inter_df.iloc[x, y]
            )
        # Index labels dataframe
        interaction_labels_df.index = shap_values_inter_df.index
        # Column labels
        interaction_labels_df.columns = shap_values_inter_df.columns
        # Reindex to sorted index
        interaction_labels_sort_df = interaction_labels_df.reindex(
            shap_values_se_mean_sort.index
        )
        # Reorder columns to sorted index
        interaction_labels_sort_df = interaction_labels_sort_df.loc[
            :, shap_values_se_mean_sort.index
        ]

        # --- Plot interaction values ---
        # Create figure
        fig, ax = plt.subplots(
            figsize=(
                x_names_max_len * 0.15 + x_names_count * 1.5 + 1.5,
                x_names_max_len * 0.15
                + x_names_count * 1.5
                + title_lines_count * 0.4
                + 0.5,
            )
        )
        # Make colorbar string
        clb_str = "mean(|SHAP values|)"
        # Plot confusion matrix
        sns.heatmap(
            shap_values_inter_sort_df,
            vmin=None,
            vmax=None,
            cmap="Greys",
            center=None,
            robust=True,
            annot=interaction_labels_sort_df,
            fmt="",
            annot_kws={"size": 10},
            linewidths=1,
            linecolor="#999999",
            cbar=True,
            cbar_kws={"label": clb_str, "shrink": 0.6},
            square=True,
            xticklabels=True,
            yticklabels=True,
            mask=None,
            ax=ax,
        )
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel(ax.get_xlabel(), fontsize=10)
        # Set x ticks size
        plt.xticks(rotation=90, fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(rotation=0, fontsize=10)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel(clb_str, fontsize=10)
        cb_ax.set_box_aspect(50)

        # --- Save plot ---
        # Make save path
        save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_4_{c_class}_avg_interaction_values"[  # noqa
            :150
        ]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # If save as svg
        if task["AS_SVG"]:
            # Save figure
            plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        # Show figure
        plt.show()


def plot_single_shap_interaction_values_dependences(
    task: dict,
    results: dict,
    store_path: str,
    DIVIDE_DATA_BY: str,
    RULE: Callable,
) -> None:
    """
    Plot single SHAP interaction values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    DIVIDE_DATA_BY: str
        Column to divida data on.
    RULE: str
        Rule to divide data.

    Returns
    -------
    None

    Raises
    ------
    ValueError: If OBJECTIVE is not regression and classification.
    """
    # Check if explanations is in results
    if "explanations" not in results.keys():
        # Log warning
        logging.warning("No explanations found in results. Skip plot.")
        # Return
        return
    # If regression
    if task["OBJECTIVE"] == "regression":
        # If no interactions
        if len(results["explanations"][0].shape) == 2:
            # Log warning
            logging.warning("No SHAP interaction values found. Skip plot.")
            # Return
            return
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # If no interactions
        if len(results["explanations"][0].shape) == 3:
            # Log warning
            logging.warning("No SHAP interaction values found. Skip plot.")
            # Return
            return

    # --- Classes ---
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Set n_classes to 1
        n_classes = 1
    # If classification
    elif task["OBJECTIVE"] == "classification":
        # Set n_classes
        n_classes = results["explanations"][0].shape[-1]
    else:
        # Raise error
        raise ValueError(f"OBJECTIVE is {task['OBJECTIVE']}.")

    # --- Plot shap values ---
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_single_shap_values(
            task,
            results["explanations"],
            c_class,
            DIVIDE_DATA_BY,
            RULE,
        )

        # --- Print shap value interaction dependencies ---
        count = 0
        for i in range(shap_values.shape[1]):
            for k in range(shap_values.shape[2]):
                # Make figure
                fig, ax = plt.subplots(figsize=(8, 5))
                # Make title string
                title_str = (
                    f"{task['ANALYSIS_NAME']}"
                    + f"\nSHAP values: single effects on {task['y_name']}"
                    + f"\nSHAP values: change from expected value of {np.round(base, decimals=2)}"  # noqa
                )
                # Add class if multiclass
                if task["OBJECTIVE"] == "classification":
                    # Make title string
                    title_str = title_str = f"{title_str} (log odds)\n class: {c_class}"
                # Plot SHAP Scatter plot
                scatter(
                    shap_values[:, i, k],
                    color=shap_values[:, k, k],
                    hist=True,
                    axis_color="#333333",
                    dot_size=16,
                    x_jitter="auto",
                    alpha=0.5,
                    title=title_str,
                    xmin=None,
                    xmax=None,
                    ymin=None,
                    ymax=None,
                    overlay=None,
                    ax=ax,
                    show=False,
                )
                # Get the current figure and axes objects.
                _, ax = plt.gcf(), plt.gca()
                # Set title size
                ax.title.set_size(10)
                # Set x label size
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                # Set x ticks size
                plt.xticks(fontsize=10)
                # Make y label
                y_label = f"Effects on {task['y_name']} (SHAP values)"
                # Set y label size
                plt.ylabel(y_label, fontsize=10)
                # Set y ticks size
                plt.yticks(fontsize=10)
                # Check if mor than 1 axes are present
                if len(fig.axes) > 1:
                    # Get colorbar
                    cb_ax = fig.axes[1]
                    # Modifying color bar tick size
                    cb_ax.tick_params(labelsize=10)
                    # Modifying color bar fontsize
                    cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=10)

                # --- Save plot ---
                # Make save path
                save_path = f"{store_path}/{task['ANALYSIS_NAME']}_{task['y_name']}_5_{c_class}_{count}_single_shap_values_dependency_{shap_values.feature_names[i]}_{shap_values.feature_names[k]}"[  # noqa
                    :150
                ]
                # Save figure
                plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
                # If save as svg
                if task["AS_SVG"]:
                    # Save figure
                    plt.savefig(f"{save_path}.svg", bbox_inches="tight")
                # Increment count
                count += 1
                # Show figure
                plt.show()


def main() -> None:
    """
    Main function of plotting.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError: If load task failed.
    FileNotFoundError: If load results failed.
    OSError: If create store directory failed.
    OSError: If save pip requirements to store failed.
    OSError: If copy iml_3_plt script to store failed.
    ValueError: If OBJECTIVE not found.
    OSError: If copy log file to store failed.
    """

    ####################################################################################
    # Script Configuration
    ####################################################################################

    # Do multiple comparison correction. bool (default: False)
    MCC = False
    # Save plots additionally as svg. bool (default: False)
    AS_SVG = False
    # Load prefix (where results are loaded from). str
    LOAD_PREFIX = "iml_2_mdl_"
    # Store prefix (where results go). str
    STORE_PREFIX = "iml_3_plt_"
    # Store suffix (to differenciate where results go). str
    STORE_SUFFIX = ""
    # Column to divida data on. str (default: "")
    DIVIDE_DATA_BY = ""
    # Rule to divide data. lambda (default: lambda item: True)
    RULE = lambda item: True  # noqa

    ####################################################################################

    # --- Load result paths ---
    res_paths = [
        f.name for f in os.scandir(".") if f.is_dir() and f.name.startswith(LOAD_PREFIX)
    ]

    # --- Loop over result paths ---
    for res_path in res_paths:
        # Get task paths
        task_file_paths = [
            f.name
            for f in os.scandir(f"./{res_path}/")
            if f.name.endswith("_task.pickle")
        ]
        # Get result paths
        results_file_paths = [
            f.name
            for f in os.scandir(f"./{res_path}/")
            if f.name.endswith("_results.pickle")
        ]

        # --- Loop over tasks ---
        for i_task, task_path in enumerate(task_file_paths):

            # --- Configure logging ---
            # Make log filename
            log_filename = f"{STORE_PREFIX}{task_path.removeprefix(LOAD_PREFIX).removesuffix('_task.pickle')}{STORE_SUFFIX}.log"  # noqa
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
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            # Set the console log format
            console_handler.setFormatter(console_formatter)
            # Set the console log level
            console_handler.setLevel(logging.INFO)
            # Add the console handler to the root logger
            logging.getLogger().addHandler(console_handler)
            logging.info(
                f"Interpretable Machine Learning - Plotting (PLT) of {res_path} started."  # noqa
            )

            # --- Load task and results ---
            try:
                # Load task description
                with open(f"{res_path}/{task_path}", "rb") as filehandle:
                    # Load task from binary data stream
                    task = pkl.load(filehandle)
            except FileNotFoundError as e:
                # Raise error
                raise e
            # Add multiple comparison correction to task
            task["MCC"] = MCC
            # Add as svg to task
            task["AS_SVG"] = AS_SVG
            try:
                # Load results
                with open(
                    f"{res_path}/{results_file_paths[i_task]}", "rb"
                ) as filehandle:
                    # Load results from binary data stream
                    results = pkl.load(filehandle)
            except FileNotFoundError as e:
                # Raise error
                raise e

            # --- Create store directory ---
            # Make store path (where plots go)
            store_path = f"{STORE_PREFIX}{res_path.removeprefix(LOAD_PREFIX)}{STORE_SUFFIX}/{task['y_name']}"  # noqa
            try:
                # Create plots directory
                os.makedirs(store_path, exist_ok=True)  # Supress FileExistsError
            except OSError as e:
                # Raise error
                raise e

            # --- Pip requirements ---
            try:
                # Get pip requirements
                pip_requirements = get_pip_requirements()
                # Open file in write mode
                with open(
                    f"{store_path}/{STORE_PREFIX}pip_requirements{STORE_SUFFIX}.txt",
                    "w",
                ) as file:
                    # Write pip requirements
                    file.write(pip_requirements)
            except OSError as e:
                # Raise error
                raise e

            # --- Python script ---
            try:
                # Copy iml_3_plt script to store path
                shutil.copy("iml_3_plt.py", f"{store_path}/iml_3_plt.py")
            except OSError as e:
                # Raise error
                raise e

            # --- Plot parameter distributions ---
            logging.info("Plot parameter distribution.")
            plot_parameter_distributions(task, results, store_path)

            # --- Plot model fit ---
            # If regression
            if task["OBJECTIVE"] == "regression":
                logging.info("Plot model fit.")
                # Print model fit as scatter plot
                plot_regression_scatter(task, results, store_path, DIVIDE_DATA_BY, RULE)
                # Print model fit as violinplot of metrics
                plot_regression_violin(task, results, store_path, DIVIDE_DATA_BY, RULE)
            # If classification
            elif task["OBJECTIVE"] == "classification":
                logging.info("Plot model fit.")
                # Print model fit as confusion matrix
                plot_classification_confusion(
                    task, results, store_path, DIVIDE_DATA_BY, RULE
                )
                # Print model fit as violinplot of metrics
                plot_classification_violin(
                    task, results, store_path, DIVIDE_DATA_BY, RULE
                )
            else:
                # Raise error
                raise ValueError("OBJECTIVE not found.")

            # --- Plot average SHAP values ---
            logging.info("Plot average SHAP values.")
            plot_avg_shap_values(task, results, store_path, DIVIDE_DATA_BY, RULE)

            # --- Plot average SHAP values distribution ---
            logging.info("Plot average SHAP values distribution.")
            plot_avg_shap_values_distributions(
                task, results, store_path, DIVIDE_DATA_BY, RULE
            )

            # --- Plot single SHAP values ---
            logging.info("Plot single SHAP values.")
            plot_single_shap_values(task, results, store_path, DIVIDE_DATA_BY, RULE)

            # --- Plot single SHAP values dependencies ---
            logging.info("Plot single SHAP values dependencies.")
            plot_single_shap_values_dependences(
                task, results, store_path, DIVIDE_DATA_BY, RULE
            )

            # --- Plot average SHAP values interactions ---
            logging.info("Plot average SHAP values interactions.")
            plot_average_shap_interaction_values(
                task, results, store_path, DIVIDE_DATA_BY, RULE
            )

            # --- Plot single SHAP values interactions ---
            logging.info("Plot single SHAP values interactions.")
            plot_single_shap_interaction_values_dependences(
                task, results, store_path, DIVIDE_DATA_BY, RULE
            )

            # --- Save log file to results directory ---
            # Log success
            logging.info(
                f"Interpretable Machine-Learning - Plotting (PLT) of {store_path} finished."  # noqa
            )
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
