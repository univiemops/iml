# -*- coding: utf-8 -*-
'''
Interpretable Machine-Learning - Plotting (PLT)
v321
@author: Dr. David Steyrl david.steyrl@univie.ac.at
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy.stats import t
from shap import Explanation
from shap.plots import beeswarm
from shap.plots import scatter
from sklearn.metrics import confusion_matrix


def lfp(path_load):
    '''
    Returns pickle file at load path.

    Parameters
    ----------
    path_load : string
        Path to pickle file.

    Returns
    -------
    data : pickle
        Returns stored data.
    '''

    # Load from pickle file ---------------------------------------------------
    # Load
    with open(path_load, 'rb') as filehandle:
        # Load data from binary data stream
        data = pkl.load(filehandle)

    # Return data -------------------------------------------------------------
    return data


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

    # Create dir of not existing ----------------------------------------------
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)

    # Return None -------------------------------------------------------------
    return


def corrected_std(differences, n_tst_over_n_trn=0.25):
    '''
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
    '''

    # Get corrected std -------------------------------------------------------
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1/kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)

    # Return corrected standard deviation -------------------------------------
    return corrected_std


def corrected_ttest(differences, n_tst_over_n_trn=0.25):
    '''
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
    '''

    # Compute t statistics and p value ----------------------------------------
    # Get mean of differences
    mean = np.mean(differences)
    # Get corrected standard deviation, make sure std is not exactly zero
    std = max(1e-6, corrected_std(differences, n_tst_over_n_trn))
    # Compute t statistics
    t_stat = mean / std
    # Compute p value for one-tailed t-test
    p_val = t.sf(t_stat, df=len(differences)-1)

    # Return t statistics and p value -----------------------------------------
    return t_stat, p_val


def print_parameter_distributions(task, results, plots_path):
    '''
    Print model parameter distributions in histogram.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Get params
    params = pd.DataFrame(results['best_params'])

    # Make plot ---------------------------------------------------------------
    # Iterate over columns of params dataframe
    for idx, (name, data) in enumerate(params.items()):
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(
            data=data.astype('float'),
            bins=30,
            kde=True,
            color='#777777',
            log_scale=True if name.endswith('path_smooth') else False,
            ax=ax)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel('Count')
        # Set title
        ax.set_title(
            task['ANALYSIS_NAME']+'\n' +
            'Parameter distribution of predicting'+' ' +
            task['y_name'],
            fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            plots_path+'/' +
            task['ANALYSIS_NAME']+'_' +
            task['y_name']+'_' +
            '0'+'_' +
            str(idx)+'_' +
            'parameter'+'_' +
            name)[:150]
        # Save figure
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()

    # Return None -------------------------------------------------------------
    return


def print_regression_scatter(task, results, plots_path):
    '''
    Print model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = np.concatenate([i['y_true'] for i in results['scores']])
    # Predicted values
    pred_values = np.concatenate([i['y_pred'] for i in results['scores']])

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(
        pred_values,
        true_values,
        zorder=2,
        alpha=0.1,
        color='#444444')
    # Add optimal fit line
    ax.plot(
        [-10000, 10000], [-10000, 10000],
        color='#999999',
        zorder=3,
        linewidth=2,
        alpha=0.3)
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove ticks
    ax.tick_params(
        axis='both',
        which='major',
        reset=True,
        bottom=True,
        top=False,
        left=True,
        right=False)
    # Add grid
    ax.grid(visible=True, which='major', axis='both')
    # Modify grid
    ax.tick_params(grid_linestyle=':', grid_alpha=.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(
        min(true_values) - true_values_range/20,
        max(true_values) + true_values_range/20)
    # Set y-axis limits
    ax.set_ylim(
        min(true_values) - true_values_range/20,
        max(true_values) + true_values_range/20)
    # Set title
    ax.set_title(
        task['ANALYSIS_NAME']+'\n' +
        'Performance of predicting'+' ' +
        task['y_name'],
        fontsize=10)
    # Set xlabel
    ax.set_xlabel('Predicted values', fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel('True values', fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # Add MAE -----------------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh)-np.array(mae))
    # Add original outcome MAE results to plot
    ax.text(
        .35, .09,
        ('Original outcome: MAE mean'+r'$\pm$'+'std {:.2f}'+r'$\pm$' +
         '{:.2f} | med {:.2f}').format(
             np.mean(mae),
             np.std(mae),
             np.median(mae)),
        transform=ax.transAxes,
        fontsize=10)
    # Add shuffled outcome MAE results to plot
    ax.text(
        .35, .055,
        ('Shuffled outcome: MAE mean'+r'$\pm$'+'std {:.2f}'+r'$\pm$' +
         '{:.2f} | med {:.2f}').format(
            np.mean(mae_sh),
            np.std(mae_sh),
            np.median(mae_sh)),
        transform=ax.transAxes,
        fontsize=10)
    # Make pval string
    if pval_mae <= 0.001:
        pval_string = 'p\u22640.001'
    else:
        pval_string = 'p={:.3f}'.format(pval_mae)
    # Add p value to the plot
    ax.text(
        .35, .02,
        ('Original vs. shuffled outcome: ')+pval_string,
        transform=ax.transAxes,
        fontsize=10)

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2)-np.array(r2_sh))
    # Add original outcome R² results to plot
    ax.text(
        .02, .96,
        ('Original outcome: R² mean'+r'$\pm$'+'std {:.3f}'+r'$\pm$' +
         '{:.3f} | med {:.3f}').format(
             np.mean(r2),
             np.std(r2),
             np.median(r2)),
        transform=ax.transAxes,
        fontsize=10)
    # Add shuffled outcome R² results to plot
    ax.text(
        .02, .925,
        ('Shuffled outcome: R² mean'+r'$\pm$'+'std {:.3f}'+r'$\pm$' +
         '{:.3f} | med {:.3f}').format(
             np.mean(r2_sh),
             np.std(r2_sh),
             np.median(r2_sh)),
        transform=ax.transAxes,
        fontsize=10)
    # Make pval string
    if pval_r2 <= 0.001:
        pval_string = 'p\u22640.001'
    else:
        pval_string = 'p={:.3f}'.format(pval_r2)
    # Add p value to the plot
    ax.text(
        .02, .89,
        ('Original vs. shuffled outcome: ')+pval_string,
        transform=ax.transAxes,
        fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (
        plots_path+'/' +
        task['ANALYSIS_NAME']+'_' +
        task['y_name']+'_' +
        '1'+'_' +
        '0'+'_' +
        'performance')[:150]
    # Save figure
    plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()

    # Return None -------------------------------------------------------------
    return


def print_regression_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae)),
         'R²': pd.Series(np.array(r2)),
         'Data': pd.Series(['original' for _ in mae]),
         'Dummy': pd.Series(np.ones(np.array(mae).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae_sh)),
         'R²': pd.Series(np.array(r2_sh)),
         'Data': pd.Series(['shuffled' for _ in mae_sh]),
         'Dummy': pd.Series(np.ones(np.array(mae_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Mean Absolute Error', 'R²']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics)*1+1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {'original': '#777777', 'shuffled': '#eeeeee'}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(
            x=metric,
            y='Dummy',
            hue='Data',
            data=all_scores_df,
            bw_method='scott',
            bw_adjust=0.5,
            cut=2,
            density_norm='width',
            gridsize=100,
            width=0.8,
            inner='box',
            orient='h',
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis='both',
            which='major',
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(
            axis='y',
            color='#bbbbbb',
            linestyle='dotted',
            alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+'\n' +
        'Performance of predicting'+' ' +
        task['y_name'])
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (
        plots_path+'/' +
        task['ANALYSIS_NAME']+'_' +
        task['y_name']+'_' +
        '1'+'_' +
        '1'+'_' +
        'performance_distribution')[:150]
    # Save figure
    plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()

    # Return None -------------------------------------------------------------
    return


def print_classification_confusion(task, results, plots_path):
    '''
    Print model fit as confusion matrix (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.
    class_tag : list
        List of class_tags

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = [i['y_true'] for i in results['scores']]
    # Predicted values
    pred_values = [i['y_pred'] for i in results['scores']]
    # Accuracy
    acc = [i['acc'] for i in results['scores']]
    # Schuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values)).tolist()

    # Get count confusion matrix ----------------------------------------------
    # Loop over single results
    # for true, pred, w in zip(true_values, pred_values, sample_weights):
    for true, pred in zip(true_values, pred_values):
        if 'con_mat_count' not in locals():
            # Compute confusion matrix
            con_mat_count = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize=None)
        else:
            # Add confusion matrix
            con_mat_count = np.add(
                con_mat_count,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize=None))
    # Get normalized confusion matrix -----------------------------------------
    # Loop over single results
    # for true, pred, w in zip(true_values, pred_values, sample_weights):
    for true, pred in zip(true_values, pred_values):
        if 'con_mat' not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize='true')
        else:
            # Add confusion matrix
            con_mat = np.add(
                con_mat,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize='true'))
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(con_mat.shape[0]*.5+4,
                 con_mat.shape[0]*.5+3.5))
    # Use tight layout
    plt.tight_layout()
    # Plot count confusion matrix
    sns.heatmap(
        con_mat_count,
        vmin=None,
        vmax=None,
        cmap='Greys',
        center=None,
        robust=True,
        annot=True,
        fmt='',
        annot_kws={'size': 10},
        linewidths=1,
        linecolor='#999999',
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[0])
    # Add title to plot
    ax[0].set_title('# predictions', fontsize=10)
    # Add x label to plot
    ax[0].set_xlabel('Predicted class', fontsize=10)
    # Add y label to plot
    ax[0].set_ylabel('True class', fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[0].tick_params(axis='y', labelsize=10, labelrotation=0)
    # Plot normalized confusion matrix
    sns.heatmap(
        con_mat_norm*100,
        vmin=None,
        vmax=None,
        cmap='Greys',
        center=None,
        robust=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 10},
        linewidths=1,
        linecolor='#999999',
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[1])
    # Add title to plot
    ax[1].set_title('norm. to True class', fontsize=10)
    # Add x label to plot
    ax[1].set_xlabel('Predicted class', fontsize=10)
    # Add y label to plot
    ax[1].set_ylabel('True class', fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[1].tick_params(axis='y', labelsize=10, labelrotation=0)
    # Calculate p-value of accuracy and shuffle accuracy
    tstat_acc, pval_acc = corrected_ttest(np.array(acc)-np.array(acc_sh))
    # Make pval string
    if pval_acc <= 0.001:
        pval_string = 'p\u22640.001'
    else:
        pval_string = 'p={:.3f}'.format(pval_acc)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+'\n' +
        'Performance of predicting'+' ' +
        task['y_name']+'\n' +
        'Original outcome balanced acc: mean'+r'$\pm$'+'std {:.2f}' +
        r'$\pm$'+'{:.2f} | med {:.2f}'+'\n' +
        'Shuffled outcome balanced acc: mean'+r'$\pm$'+'std {:.2f}' +
        r'$\pm$'+'{:.2f} | med {:.2f}'+'\n' +
        'Original vs. shuffled outcome: ').format(
        np.mean(acc)*100,
        np.std(acc)*100,
        np.median(acc)*100,
        np.mean(acc_sh)*100,
        np.std(acc_sh)*100,
        np.median(acc_sh)*100)+pval_string+'\n'
    # Set title
    plt.suptitle(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (
        plots_path+'/' +
        task['ANALYSIS_NAME']+'_' +
        task['y_name']+'_' +
        '1'+'_' +
        '0'+'_' +
        'performance')[:150]
    # Save figure
    plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()

    # Return None -------------------------------------------------------------
    return


def print_classification_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract accuracy
    acc = [i['acc'] for i in results['scores']]
    # Extract shuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc)),
         'Data': pd.Series(['original' for _ in acc]),
         'Dummy': pd.Series(np.ones(np.array(acc).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc_sh)),
         'Data': pd.Series(['shuffled' for _ in acc_sh]),
         'Dummy': pd.Series(np.ones(np.array(acc_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Accuracy']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics)*1+1))
    # Make color palette
    mypal = {'original': '#777777', 'shuffled': '#eeeeee'}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(
            x=metric,
            y='Dummy',
            hue='Data',
            data=all_scores_df,
            bw_method='scott',
            bw_adjust=0.5,
            cut=2,
            density_norm='width',
            gridsize=100,
            width=0.8,
            inner='box',
            orient='h',
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis='both',
            which='major',
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(
            axis='y',
            color='#bbbbbb',
            linestyle='dotted',
            alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+'\n' +
        'Performance of predicting'+' ' +
        task['y_name'])
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (
        plots_path+'/' +
        task['ANALYSIS_NAME']+'_' +
        task['y_name']+'_' +
        '1'+'_' +
        '1'+'_' +
        'performance_distribution')[:150]
    # Save figure
    plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()

    # Return None -------------------------------------------------------------
    return


def get_shap_effects(task, explainations, c_class=-1):
    '''
    Get SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_effects : list
        SHAP effects.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values[:, :, c_class]), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(
            k.values[:, :, :, c_class], axis=2)), axis=0)
            for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, :, c_class].base_values
                                  for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Make SHAP effects dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])

    # Return shap effects -----------------------------------------------------
    return shap_effects_df, base


def print_shap_effects(task, results, plots_path):
    '''
    Print SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task,
            results['explainations'],
            c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(
            task,
            results['explainations_sh'],
            c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=True)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make horizontal bar plot
        shap_effects_se_mean_sort.plot(
            kind='barh',
            figsize=(x_names_max_len*.1+8, x_names_count*.4+1),
            color='#777777',
            fontsize=10)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+'\n' +
            'mean(|SHAP values|): mean absolute marginal change in ' +
            task['y_name']+'\n' +
            ' from it\'s expected value of ' +
            str(np.round(base, decimals=2)))
        # Add class if multiclass
        if task['OBJECTIVE'] == 'multiclass':
            # Make title string
            title_str = title_str+'\n class: '+str(c_class)
        # Set title
        ax.set_title(title_str, fontsize=10)

        # Compute SHAP effect p values ----------------------------------------
        # Init p value list
        pval = []
        # Iterate over predictors
        for pred_name, pred_data in shap_effects_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy()-shap_effects_sh_df[pred_name].to_numpy())
            # Add to pval list
            pval.append(np.around(c_pval, decimals=3))
        # Make pval series
        pval_se = pd.Series(data=pval, index=task['x_names'])
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval_se = pval_se*x_names_count
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)

        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make pval string
            if pval_se[c_pred] <= 0.001:
                pval_string = 'p\u22640.001'
            else:
                pval_string = 'p={:.3f}'.format(pval_se[c_pred])
            # Make text string
            txt_str = ('{:.2f}'+' | '+pval_string).format(c_val)
            # Add values to plot
            ax.text(
                c_val+shap_effects_se_mean.max()*0.01,
                i,
                txt_str,
                color='k',
                va='center',
                fontsize=10)
        # Get x limits
        x_left, x_right = plt.xlim()
        # Set x limits
        plt.xlim(x_left, x_right + x_right*.15)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (
            plots_path+'/' +
            task['ANALYSIS_NAME']+'_' +
            task['y_name']+'_' +
            '2'+'_' +
            '0'+'_' +
            str(c_class)+'_' +
            'shap_effect')[:150]
        # Save figure
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()

    # Return None -------------------------------------------------------------
    return


def print_shap_effects_distribution(task, results, plots_path):
    '''
    Print SHAP values distribution.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap effects distribution ------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task,
            results['explainations'],
            c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(
            task,
            results['explainations_sh'],
            c_class)

        # Process SHAP effects-------------------------------------------------
        # Sorting index by mean value of columns
        i_srt = shap_effects_df.mean().sort_values(ascending=False).index
        # Sort SHAP effects dataframe
        shap_effects_df_sort = shap_effects_df.reindex(i_srt, axis=1)
        # Sort shuffle SHAP effects dataframe
        shap_effects_sh_df_sort = shap_effects_sh_df.reindex(i_srt, axis=1)
        # Add data origin to SHAP effects dataframe
        shap_effects_df_sort['Data'] = pd.DataFrame(
            ['original' for _ in range(shap_effects_df_sort.shape[0])],
            columns=['Data'])
        # Add data origin to shuffle SHAP effects dataframe
        shap_effects_sh_df_sort['Data'] = pd.DataFrame(
            ['shuffled' for _ in range(shap_effects_sh_df_sort.shape[0])],
            columns=['Data'])
        # Get value name
        value_name = 'mean(|SHAP value|)'
        # Melt SHAP effects dataframe
        shap_effects_df_sort_melt = shap_effects_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Melt shuffle SHAP effects dataframe
        shap_effects_sh_df_sort_melt = shap_effects_sh_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Concatenate importances dataframes
        shap_effects_df_sort_melt_all = pd.concat([
            shap_effects_df_sort_melt,
            shap_effects_sh_df_sort_melt], axis=0)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make figure
        fig, ax = plt.subplots(
            figsize=(x_names_max_len*.1+8,
                     x_names_count*.4+1))
        # Make color palette
        mypal = {'original': '#777777', 'shuffled': '#eeeeee'}
        # Plot data
        sns.violinplot(
            x=value_name,
            y='predictors',
            hue='Data',
            data=shap_effects_df_sort_melt_all,
            bw_method='scott',
            bw_adjust=0.5,
            cut=2,
            density_norm='width',
            gridsize=100,
            width=0.8,
            inner='box',
            orient='h',
            linewidth=.5,
            saturation=1,
            ax=ax,
            palette=mypal)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel('', fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Set legend position
        plt.legend(loc='lower right')
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+'\n' +
            'mean(|SHAP values|): mean absolute marginal change in ' +
            task['y_name']+'\n' +
            ' from it\'s expected value of ' +
            str(np.round(base, decimals=2)))
        # Add class if multiclass
        if task['OBJECTIVE'] == 'multiclass':
            # Make title string
            title_str = title_str+'\n class: '+str(c_class)
        # Add title
        ax.set_title(title_str, fontsize=10)

        # Save plots and results ----------------------------------------------
        # Make save path
        save_path = (
            plots_path+'/' +
            task['ANALYSIS_NAME']+'_' +
            task['y_name']+'_' +
            '2'+'_' +
            '1'+'_' +
            str(c_class)+'_' +
            'shap_effect_distribution')[:150]
        # Save figure
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show figure
        plt.show()

    # Return None -------------------------------------------------------------
    return


def get_shap_values(task, explainations, c_class=-1):
    '''
    Get SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_explainations : shap explaination object
        Explaination object with SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k[:, :, :, c_class].base_values
                                  for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Return shap effects -----------------------------------------------------
    return shap_explainations, base


def print_shap_values(task, results, plots_path):
    '''
    Plot SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(
            task,
            results['explainations'],
            c_class)

        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot SHAP values beeswarm -------------------------------------------
        beeswarm(
            shap_explainations,
            max_display=len(task['x_names']),
            order=Explanation.abs.mean(0),
            clustering=None,
            cluster_threshold=0.5,
            color=None,
            axis_color='#333333',
            alpha=.66,
            show=False,
            log_scale=False,
            color_bar=True,
            plot_size=(x_names_max_len*.1+8, x_names_count*.4+1),
            color_bar_label='Predictor value')
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('SHAP values', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+'\n' +
            'SHAP values: marginal change in ' +
            task['y_name']+'\n' +
            ' from it\'s expected value of ' +
            str(np.round(base, decimals=2)))
        # Add class if multiclass
        if task['OBJECTIVE'] == 'multiclass':
            # Make title string
            title_str = title_str+'\n class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel('Predictor value', fontsize=10)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (
            plots_path+'/' +
            task['ANALYSIS_NAME']+'_' +
            task['y_name']+'_' +
            '2'+'_' +
            '2'+'_' +
            str(c_class)+'_' +
            'shap_values')[:150]
        # Save figure
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()

    # Return None -------------------------------------------------------------
    return


def print_shap_dependences(task, results, plots_path):
    '''
    Plot SHAP dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(
            task,
            results['explainations'],
            c_class)
        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Print shap values dependencies --------------------------------------
        # Loop over predictors
        for idx, c_pred in enumerate(shap_explainations.feature_names):
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+'\n' +
                'SHAP values: marginal change in ' +
                task['y_name']+'\n' +
                ' from it\'s expected value of ' +
                str(np.round(base, decimals=2)))
            # Add class if multiclass
            if task['OBJECTIVE'] == 'multiclass':
                # Make title string
                title_str = title_str+'\n class: '+str(c_class)
            # Plot SHAP Scatter plot
            scatter(
                shap_explainations[:, idx],
                color='#777777',
                hist=True,
                axis_color='#333333',
                dot_size=16,
                x_jitter='auto',
                alpha=.5,
                title=title_str,
                xmin=None,
                xmax=None,
                ymin=None,
                ymax=None,
                overlay=None,
                ax=ax,
                show=False)
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set title size
            ax.title.set_size(10)
            # Set x label size
            plt.xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Make y label
            y_label = 'Marginal SHAP values'
            # Set y label size
            plt.ylabel(y_label, fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            save_path = (
                plots_path+'/' +
                task['ANALYSIS_NAME']+'_' +
                task['y_name']+'_' +
                '3'+'_' +
                str(c_class)+'_' +
                str(idx)+'_' +
                'dependency'+'_' +
                c_pred)[:150]

            # Save figure
            plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure
                plt.savefig(save_path+'.svg',  bbox_inches='tight')
            # Show figure
            plt.show()

    # Return None -------------------------------------------------------------
    return


def get_shap_effects_inter(task, explainations, c_class=-1):
    '''
    Get SHAP based global interaction effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_effects : list
        SHAP effects.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap interaction effects --------------------------------------------
    # Case 1: interaction and regression
    if task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Get SHAP interaction effects
        shap_effects_inter = np.array([np.mean(np.abs(k.values), axis=0)
                                       for k in explainations])
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Get SHAP interaction effects
        shap_effects_inter = np.array([np.mean(np.abs(k.values), axis=0)
                                       for k in explainations])
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Get SHAP interaction effects
        shap_effects_inter = np.array([np.mean(np.abs(
            k[:, :, :, c_class].values), axis=0) for k in explainations])
        # Base value
        base = np.mean(np.hstack([k[:, :, :, c_class].base_values
                                  for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Return shap interaction effects -----------------------------------------
    return shap_effects_inter, base


def print_shap_effects_interactions(task, results, plots_path):
    '''
    Plot SHAP effects inclusive interactions.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task,
            results['explainations'],
            c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=False)

        # Get SHAP effects interactions ---------------------------------------
        # SHAP effects
        shap_effects_inter, base_inter = get_shap_effects_inter(
            task,
            results['explainations'],
            c_class)
        # Make dataframe
        shap_effects_inter_df = pd.DataFrame(
            np.mean(shap_effects_inter, axis=0),
            index=task['x_names'],
            columns=task['x_names'])
        # Reindex to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_sort_df.loc[:, shap_effects_se_mean_sort.index]
        # SHAP effects shuffle
        shap_effects_inter_sh, base_inter_sh = get_shap_effects_inter(
            task,
            results['explainations_sh'],
            c_class)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Make labels with pvales ---------------------------------------------
        # Init p values
        pval = np.zeros(
            (shap_effects_inter.shape[1],
             shap_effects_inter.shape[2]))
        # Iterate over shap_effects
        for x, y in np.ndindex(
                (shap_effects_inter.shape[1],
                 shap_effects_inter.shape[2])):
            # Get current SHAP effect
            c_effect = shap_effects_inter[:, x, y]
            # Get current SHAP effect shuffle
            c_effect_sh = shap_effects_inter_sh[:, x, y]
            # Calculate p-value
            _, pval[x, y] = corrected_ttest(c_effect-c_effect_sh)
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval = pval*(x_names_count**2)
            # Set p values > 1 to 1
            pval = pval.clip(None, 1)
        # Initialize labels dataframe
        interaction_labels_df = pd.DataFrame(np.zeros([
            shap_effects_inter.shape[1],
            shap_effects_inter.shape[2]]),
            dtype='string')
        # Iterate labels
        for x, y in np.ndindex(
                (shap_effects_inter.shape[1],
                 shap_effects_inter.shape[2])):
            # Make pval string
            if pval[x, y] <= 0.001:
                pval_string = 'p\u22640.001'
            else:
                pval_string = 'p={:.3f}'.format(pval[x, y])
            # Make label
            interaction_labels_df.iloc[x, y] = (
                '{:.2f}' +
                '\n' +
                pval_string).format(shap_effects_inter_df.iloc[x, y])
        # Index labels dataframe
        interaction_labels_df.index = shap_effects_inter_df.index
        # Column labels
        interaction_labels_df.columns = shap_effects_inter_df.columns
        # Reindex to sorted index
        interaction_labels_sort_df = \
            interaction_labels_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        interaction_labels_sort_df = \
            interaction_labels_sort_df.loc[:, shap_effects_se_mean_sort.index]

        # Plot interaction effects --------------------------------------------
        # Create figure
        fig, ax = plt.subplots(
            figsize=(x_names_max_len*.15+x_names_count*1.5+1.5,
                     x_names_max_len*.15+x_names_count*1.5+1.5))
        # Make colorbar string
        clb_str = ('mean(|SHAP value|)')
        # Plot confusion matrix
        sns.heatmap(
            shap_effects_inter_sort_df,
            vmin=None,
            vmax=None,
            cmap='Greys',
            center=None,
            robust=True,
            annot=interaction_labels_sort_df,
            fmt='',
            annot_kws={'size': 10},
            linewidths=1,
            linecolor='#999999',
            cbar=True,
            cbar_kws={'label': clb_str, 'shrink': 0.6},
            square=True,
            xticklabels=True,
            yticklabels=True,
            mask=None,
            ax=ax)
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
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+'\n' +
            'mean(|SHAP values|): mean absolute partial change in ' +
            task['y_name']+'\n' +
            ' from it\'s expected value of ' +
            str(np.round(base, decimals=2)))
        # Add class if multiclass
        if task['OBJECTIVE'] == 'multiclass':
            # Make title string
            title_str = title_str+'\n class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel(clb_str, fontsize=10)
        cb_ax.set_box_aspect(50)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (
            plots_path+'/' +
            task['ANALYSIS_NAME']+'_' +
            task['y_name']+'_' +
            '4'+'_' +
            str(c_class)+'_' +
            'interaction_effect')[:150]
        # Save figure
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()

    # Return None -------------------------------------------------------------
    return


def print_shap_interaction_values(task, results, plots_path):
    '''
    Plot SHAP interaction values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If regression or binary
    if (task['OBJECTIVE'] == 'regression' or task['OBJECTIVE'] == 'binary'):
        # Set n_classes to 1
        n_classes = 1
    # Other cases
    else:
        # Set n_classes
        n_classes = results['explainations'][0].shape[-1]

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task,
                                            results['explainations'],
                                            c_class)

        # Print shap value interaction dependencies ---------------------------
        count = 0
        for i in range(shap_values.shape[1]):
            for k in range(shap_values.shape[2]):
                # Make figure
                fig, ax = plt.subplots(figsize=(8, 5))
                # Make title string
                title_str = (
                    task['ANALYSIS_NAME']+'\n' +
                    'SHAP values: partial change in ' +
                    task['y_name']+'\n' +
                    ' from it\'s expected value of ' +
                    str(np.round(np.mean(np.hstack(
                        [k.base_values for k in results['explainations']])),
                        decimals=2)))
                # Add class if multiclass
                if task['OBJECTIVE'] == 'multiclass':
                    # Make title string
                    title_str = title_str+'\n class: '+str(c_class)
                # Plot SHAP Scatter plot
                scatter(
                    shap_values[:, i, k],
                    color=shap_values[:, k, k],
                    hist=True,
                    axis_color='#333333',
                    dot_size=16,
                    x_jitter='auto',
                    alpha=.5,
                    title=title_str,
                    xmin=None,
                    xmax=None,
                    ymin=None,
                    ymax=None,
                    overlay=None,
                    ax=ax,
                    show=False)
                # Get the current figure and axes objects.
                _, ax = plt.gcf(), plt.gca()
                # Set title size
                ax.title.set_size(10)
                # Set x label size
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                # Set x ticks size
                plt.xticks(fontsize=10)
                # Make y label
                y_label = 'Partial SHAP values'
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

                # Save plot ---------------------------------------------------
                # Make save path
                save_path = (
                    plots_path+'/' +
                    task['ANALYSIS_NAME']+'_' +
                    task['y_name']+'_' +
                    '5'+'_' +
                    str(c_class)+'_' +
                    str(count)+'_' +
                    'dependeny_interaction'+'_' +
                    shap_values.feature_names[i]+'_' +
                    shap_values.feature_names[k])[:150]
                # Save figure
                plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
                # Check if save as svg is enabled
                if task['AS_SVG']:
                    # Save figure
                    plt.savefig(save_path+'.svg',  bbox_inches='tight')
                # Increment count
                count += 1
                # Show figure
                plt.show()

    # Return None -------------------------------------------------------------
    return


def main():
    '''
    Main function of plot results of machine-learning based data analysis.

    Returns
    -------
    None.
    '''

    ###########################################################################
    # Specify plot tasks
    ###########################################################################

    # Plot hyper parameter distributions. bool (default: True)
    PPD = True
    # Plot SHAP interactions. bool (default: True)
    PSI = True
    # Do multiple comparison correction. bool (default: False)
    MCC = False
    # Save plots additionally as svg. bool (default: False)
    AS_SVG = False

    ###########################################################################

    # Load result paths -------------------------------------------------------
    res_paths = [f.name for f in os.scandir('.')
                 if f.is_dir() and f.name.startswith('res_iml_')]

    # Loop over result paths --------------------------------------------------
    for res_path in res_paths:
        # Get task paths
        task_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                      if f.name.endswith('_task.pickle')]
        # Get result paths
        results_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                         if f.name.endswith('_results.pickle')]

        # Loop over tasks -----------------------------------------------------
        for i_task, task_path in enumerate(task_paths):

            # Load task and results, create plots directory -------------------
            # Load task description
            task = lfp(res_path+'/'+task_path)
            # Add multiple comparison correction to task
            task['MCC'] = MCC
            # Add as svg to task
            task['AS_SVG'] = AS_SVG
            # Load results
            results = lfp(res_path+'/'+results_paths[i_task])
            # Plots path
            plots_path = res_path+'/'+task['y_name']+'_plots'
            # Create plots dir
            create_dir(plots_path)

            # Plot parameter distributions ------------------------------------
            if PPD:
                print_parameter_distributions(task, results, plots_path)

            # Plot model fit --------------------------------------------------
            # Regressor
            if task['OBJECTIVE'] == 'regression':
                # Print model fit as scatter plot
                print_regression_scatter(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_regression_violin(task, results, plots_path)
            # Classification
            elif (task['OBJECTIVE'] == 'binary' or
                  task['OBJECTIVE'] == 'multiclass'):
                # Print model fit as confusion matrix
                print_classification_confusion(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_classification_violin(task, results, plots_path)
            # Other
            else:
                # Raise error
                raise ValueError('OBJECTIVE not found.')

            # Plot SHAP effects -----------------------------------------------
            print_shap_effects(task, results, plots_path)

            # Plot SHAP effects distribution ----------------------------------
            print_shap_effects_distribution(task, results, plots_path)

            # Plot SHAP values ------------------------------------------------
            print_shap_values(task, results, plots_path)

            # Plot SHAP dependencies ------------------------------------------
            print_shap_dependences(task, results, plots_path)

            # Plot SHAP effects interactions ----------------------------------
            if task['SHAP_INTERACTIONS'] and PSI:
                print_shap_effects_interactions(task, results, plots_path)

            # Plot SHAP interaction values ------------------------------------
            if task['SHAP_INTERACTIONS'] and PSI:
                print_shap_interaction_values(task, results, plots_path)

    # Return None -------------------------------------------------------------
    return


if __name__ == '__main__':
    main()
