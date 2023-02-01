# -*- coding: utf-8 -*-
'''
Plot results of machine learning based data analysis
v218
@author: Dr. David Steyrl david.steyrl@gmail.com
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from itertools import permutations
from scipy.stats import t
from shap import dependence_plot
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
    # Load from pickle file
    with open(path_load, 'rb') as filehandle:
        # Load data from binary data stream
        data = pickle.load(filehandle)
    # Return loaded data
    return data


def create_dir(path):
    '''
    Create specified directiry if not existing.

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


def corr_std_error(x, tst_size_frac):
    '''Corrects standard error using Nadeau and Bengio's approach.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366

    Parameters
    ----------
    x : array of float
        Cross validation result data.
    tst_size_frac : float
        Fraction of samples in the testing set.

    Returns
    -------
    corr_std_error : float
        Variance-corrected standard error of the data x.
    '''
    # Compute variance of x
    var_x = np.var(x)
    # Compute corrected standard error
    corr_std_error = np.sqrt(var_x/len(x) +
                             var_x*tst_size_frac/(1-tst_size_frac))
    # Return corrected std
    return corr_std_error


def dep_two_sample_ttest(x1, x2, tst_size_frac, side='two'):
    '''
    Implementation of the Nadeau and Bengio correction of dependent sample
    (due to cross-validation's resampling) two sample Student's t-test for
    unequal sample sizes and unequal variances aka Welch's test
    t_stat = mean(x1)-mean(x2)/sqrt(std_error(x1)²+std_error(x2)²)
    whereas std_error(x) (= std(x)/sqrt(n_samples(x))) is replaced by
    corrected_std_error (= corrected_std(x)/sqrt(n_samples(x)))
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    x1 : ndarray
        Data of condition 1.
    x2 : ndarray
        Data of condition 2.
    tst_size_frac : float
        Test size fraction.
    side : sring ('one', 'two')
        If test is one or two sided. Default is two.

    Returns
    -------
    t_stat : float
        Corrected t statistic.
    df : float
        Degrees of freedom.
    p : float
        Corrected p value.

    '''
    # Get n samples data 1
    n1 = len(x1)
    # Get n samples data 2
    n2 = len(x2)
    # Get corrected standard error of data 1
    std_error_x1 = corr_std_error(x1, tst_size_frac)
    # Get corrected standard error of data 2
    std_error_x2 = corr_std_error(x2, tst_size_frac)
    # Compute corrected t statistics
    t_stat = (np.mean(x1)-np.mean(x2))/np.sqrt(std_error_x1**2+std_error_x2**2)
    # Compute standard deviation of data 1
    std_dev_x1 = std_error_x1*np.sqrt(n1)
    # Compute standard deviation of data 2
    std_dev_x2 = std_error_x2*np.sqrt(n2)
    # degrees of freedom
    df = (std_dev_x1**2/n1+std_dev_x2**2/n2)**2/(
        std_dev_x1**4/(n1**2 * (n1-1)) + std_dev_x2**4/(n2**2 * (n2-1)))
    # calculate the p-value
    if 'one' in side:
        p_val = t.sf(t_stat, df)
    elif 'two' in side:
        p_val = t.sf(t_stat, df) * 2.0
    else:
        print('Error: unknown side')
    # return everything
    return t_stat, df, p_val


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
    # Get best params
    params = pd.DataFrame(results['best_params'])
    # Iterate over columns of params dataframe
    for (name, data) in params.items():
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(data=data,
                     bins=30,
                     kde=True,
                     color='#777777',
                     ax=ax)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel('Number')
        # Set title
        ax.set_title(task['ANALYSIS_NAME']+' ' +
                     task['ESTIMATOR_NAME']+' ' +
                     'parameter distribution for predicting'+' ' +
                     task['y_name'][0],
                     fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     task['ESTIMATOR_NAME']+'_' +
                     'param_dist_predicting'+'_' +
                     task['y_name'][0]+'_' +
                     name)[:135]
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()


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
    # True values
    true_values = np.concatenate([i['y_true'] for i in results['scores']])
    # Predicted values
    pred_values = np.concatenate([i['y_pred'] for i in results['scores']])
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(pred_values,
               true_values,
               zorder=2,
               alpha=0.1,
               color='#444444')
    # Add optimal fit line
    ax.plot([-10000, 10000], [-10000, 10000],
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
    ax.tick_params(axis='both', which='major', reset=True,
                   bottom=True, top=False, left=True, right=False)
    # Add grid
    ax.grid(visible=True, which='major', axis='both')
    # Modify grid
    ax.tick_params(grid_linestyle=':', grid_alpha=.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set y-axis limits
    ax.set_ylim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set title
    ax.set_title(task['ANALYSIS_NAME']+' ' +
                 task['ESTIMATOR_NAME']+' ' +
                 'predicting'+' ' +
                 task['y_name'][0],
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
    _, _, pval_mae = dep_two_sample_ttest(
        np.array(mae_sh), np.array(mae), task['TST_SIZE_FRAC'], side='one')
    # Add MAE results to plot
    ax.text(.40, .055, ('MAE original mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}').format(
            np.mean(mae),
            np.std(mae),
            np.median(mae)),
            transform=ax.transAxes,
            fontsize=8)
    # Add MAE p val results to plot
    ax.text(.40, .02, ('MAE shuffle mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}|p:{:.3f}').format(
            np.mean(mae_sh),
            np.std(mae_sh),
            np.median(mae_sh),
            pval_mae),
            transform=ax.transAxes,
            fontsize=8)

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Calculate p-value between R² and shuffle R²
    _, _, pval_r2 = dep_two_sample_ttest(
        np.array(r2), np.array(r2_sh), task['TST_SIZE_FRAC'], side='one')
    # Add R² results to plot
    ax.text(.02, .96, ('R² original mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}').format(
            np.mean(r2),
            np.std(r2),
            np.median(r2)),
            transform=ax.transAxes,
            fontsize=8)
    # Add R² p val results to plot
    ax.text(.02, .925, ('R² shuffle mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}|p:{:.3f}').format(
            np.mean(r2_sh),
            np.std(r2_sh),
            np.median(r2_sh),
            pval_r2),
            transform=ax.transAxes,
            fontsize=8)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'predicting'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


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
         'R2': pd.Series(np.array(r2)),
         'Data': pd.Series(['original' for _ in mae]),
         'Dummy': pd.Series(np.ones(np.array(mae).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae_sh)),
         'R2': pd.Series(np.array(r2_sh)),
         'Data': pd.Series(['shuffle' for _ in mae_sh]),
         'Dummy': pd.Series(np.ones(np.array(mae_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Mean Absolute Error', 'R2']
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics)*.75+1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw='scott', cut=2, scale='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
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
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'predicting'+'_' +
                 'distribution'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()


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

    Returns
    -------
    None.

    '''
    # True values
    true_values = [i['y_true'] for i in results['scores']]
    # Predicted values
    pred_values = [i['y_pred'] for i in results['scores']]
    # Sample weights  list
    sample_weights = [i['class_weights'] for i in results['scores']]
    # Accuracy
    acc = [i['acc'] for i in results['scores']]
    # Schuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values)).tolist()

    # Get confusion matrix ----------------------------------------------------
    # Loop over single results
    for true, pred, w in zip(true_values, pred_values, sample_weights):
        if 'con_mat' not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all')
        else:
            # Add confusion matrix
            con_mat = np.add(con_mat, confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all'))
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(figsize=(con_mat.shape[0]*.5+3,
                                    con_mat.shape[0]*.5+3))
    # Plot confusion matrix
    sns.heatmap(con_mat_norm*100,
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
                xticklabels=class_labels,
                yticklabels=class_labels,
                mask=None,
                ax=ax)
    # Add x label to plot
    plt.xlabel('Predicted class', fontsize=10)
    # Add y label to plot
    plt.ylabel('True class', fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    plt.yticks(rotation=0, fontsize=10)
    # Calculate p-value of accuracy and shuffle accuracy
    _, _, pval_acc = dep_two_sample_ttest(
        np.array(acc), np.array(acc_sh), task['TST_SIZE_FRAC'])
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0]+'\n' +
        'Orig. data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'Shuffle data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'p-value of orig. and shuffle: {:.3f}'+'\n').format(
        np.mean(acc)*100,
        np.std(acc)*100,
        np.median(acc)*100,
        np.mean(acc_sh)*100,
        np.std(acc_sh)*100,
        np.median(acc_sh)*100,
        pval_acc)
    # Set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'predicting'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()


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
         'Data': pd.Series(['shuffle' for _ in acc_sh]),
         'Dummy': pd.Series(np.ones(np.array(acc_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Accuracy']
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics)*.75+1))
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw='scott', cut=2, scale='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
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
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'predicting'+'_' +
                 'distribution'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()


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
    # Get SHAP effects --------------------------------------------------------
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in results['explainations']]
    # Without SHAP interaction effects
    else:
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in results['explainations']]
    # Make dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])

    # Process SHAP effects-----------------------------------------------------
    # Mean shap values
    shap_effects_se_mean = shap_effects_df.mean(axis=0)
    # Sort from highto low
    shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
        ascending=True)

    # Additional info ---------------------------------------------------------
    # x names lengths
    x_names_max_len = max([len(i) for i in task['x_names']])
    # x names count
    x_names_count = len(task['x_names'])

    # Plot --------------------------------------------------------------------
    # Make horizontal bar plot
    shap_effects_se_mean_sort.plot(
        kind='barh',
        figsize=(x_names_max_len*.1+7, x_names_count*.4+1),
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
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'SHAP effects for'+' ' +
        task['y_name'][0]+'\n' +
        'mean(|SHAP values|) = mean absolute deviation from expected value (' +
        str(np.round(np.mean(
            [k.base_values for k in results['explainations']]), decimals=2)) +
        ')')
    # Set title
    ax.set_title(title_str, fontsize=10)

    # Compute SHAP effect p values --------------------------------------------
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # Get shuffle SHAP effects
        shap_effects_sh = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                           for k in results['explainations_sh']]
    # Without SHAP interaction effects
    else:
        # Get shuffle SHAP effects
        shap_effects_sh = [np.mean(np.abs(k.values), axis=0)
                           for k in results['explainations_sh']]
    # Make dataframe
    shap_effects_sh_df = pd.DataFrame(shap_effects_sh, columns=task['x_names'])
    # Init p value list
    pval = []
    # Iterate over predictors
    for pred_name, pred_data in shap_effects_df.items():
        # Get current p value
        _, _, c_pval = dep_two_sample_ttest(
            pred_data.to_numpy(),
            shap_effects_sh_df[pred_name].to_numpy(),
            task['TST_SIZE_FRAC'],
            side='one')
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

    # Add SHAP effect values and p values as text -----------------------------
    # Loop over values
    for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
        # Make test string
        txt_str = (str(np.around(c_val, decimals=2))+'|' +
                   'p '+str(pval_se[c_pred]))
        # Add values to plot
        ax.text(c_val, i, txt_str, color='k',
                va='center', fontsize=8)
    # Get x limits
    x_left, x_right = plt.xlim()
    plt.xlim(x_left, x_right + x_right*.1)

    # Save plot ---------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'shap_effects'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


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
    # Get SHAP effects --------------------------------------------------------
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in results['explainations']]
    # Without SHAP interaction effects
    else:
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in results['explainations']]
    # Make dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # Get shuffle SHAP effects
        shap_effects_sh = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                           for k in results['explainations_sh']]
    # Without SHAP interaction effects
    else:
        # Get shuffle SHAP effects
        shap_effects_sh = [np.mean(np.abs(k.values), axis=0)
                           for k in results['explainations_sh']]
    # Make dataframe
    shap_effects_sh_df = pd.DataFrame(shap_effects_sh, columns=task['x_names'])

    # Process SHAP effects-----------------------------------------------------
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
        ['shuffle' for _ in range(shap_effects_sh_df_sort.shape[0])],
        columns=['Data'])
    # Get value name
    value_name = 'mean(|SHAP value|)'
    # Melt SHAP effects dataframe
    shap_effects_df_sort_melt = shap_effects_df_sort.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Melt shuffle SHAP effects dataframe
    shap_effects_sh_df_sort_melt = shap_effects_sh_df_sort.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Concatenate importances dataframes
    shap_effects_df_sort_melt_all = pd.concat([
        shap_effects_df_sort_melt,
        shap_effects_sh_df_sort_melt], axis=0)

    # Additional info ---------------------------------------------------------
    # x names lengths
    x_names_max_len = max([len(i) for i in task['x_names']])
    # x names count
    x_names_count = len(task['x_names'])

    # Plot --------------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(x_names_max_len*.1+7,
                                    x_names_count*.4+1))
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Plot data
    sns.violinplot(x=value_name, y='predictors', hue='Data',
                   data=shap_effects_df_sort_melt_all, bw='scott',
                   cut=2, scale='width', gridsize=100, width=0.8,
                   inner='box', orient='h', linewidth=.5,
                   saturation=1, ax=ax, palette=mypal)
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
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'SHAP effects distribution for'+' ' +
        task['y_name'][0]+'\n' +
        'mean(|SHAP values|) = mean absolute deviation from expected value (' +
        str(np.round(np.mean(
            [k.base_values for k in results['explainations']]), decimals=2)) +
        ')')
    # Add title
    ax.set_title(title_str, fontsize=10)

    # Save plots and results --------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'shap_effects_distribuation'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()


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
    # Get SHAP explainations object -------------------------------------------
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # Explainer object
        shap_explainations = Explanation(np.vstack(
            [np.sum(k.values, axis=2) for k in results['explainations']]),
            base_values=np.hstack(
                [k.base_values for k in results['explainations']]),
            data=np.vstack(
                [k.data for k in results['explainations']]),
            display_data=None,
            instance_names=None,
            feature_names=results['explainations'][0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum(
                [k.compute_time for k in results['explainations']]))
    # Without SHAP interaction effects
    else:
        # Explainer object
        shap_explainations = Explanation(np.vstack(
            [k.values for k in results['explainations']]),
            base_values=np.hstack(
                [k.base_values for k in results['explainations']]),
            data=np.vstack(
                [k.data for k in results['explainations']]),
            display_data=None,
            instance_names=None,
            feature_names=results['explainations'][0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum(
                [k.compute_time for k in results['explainations']]))

    # Additional info ---------------------------------------------------------
    # x names lengths
    x_names_max_len = max([len(i) for i in task['x_names']])
    # x names count
    x_names_count = len(task['x_names'])

    # Plot SHAP values beeswarm -----------------------------------------------
    beeswarm(shap_explainations,
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
             plot_size=(x_names_max_len*.1+7, x_names_count*.4+1),
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
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'SHAP values for'+' ' +
        task['y_name'][0]+'\n' +
        'SHAP values = deviation from expected value (' +
        str(np.round(np.mean(
            [k.base_values for k in results['explainations']]), decimals=2)) +
        ')')
    # Add title
    plt.title(title_str, fontsize=10)
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar tick size
    cb_ax.tick_params(labelsize=10)
    # Modifying color bar fontsize
    cb_ax.set_ylabel('Predictor value', fontsize=10)

    # Save plot ---------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'shap_values'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


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
    # Get SHAP explainations object -------------------------------------------
    # With SHAP interaction effects
    if task['SHAP_INTERACTIONS']:
        # Explainer object
        shap_explainations = Explanation(np.vstack(
            [np.sum(k.values, axis=2) for k in results['explainations']]),
            base_values=np.hstack(
                [k.base_values for k in results['explainations']]),
            data=np.vstack(
                [k.data for k in results['explainations']]),
            display_data=None,
            instance_names=None,
            feature_names=results['explainations'][0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum(
                [k.compute_time for k in results['explainations']]))
    # Without SHAP interaction effects
    else:
        # Explainer object
        shap_explainations = Explanation(np.vstack(
            [k.values for k in results['explainations']]),
            base_values=np.hstack(
                [k.base_values for k in results['explainations']]),
            data=np.vstack(
                [k.data for k in results['explainations']]),
            display_data=None,
            instance_names=None,
            feature_names=results['explainations'][0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum(
                [k.compute_time for k in results['explainations']]))

    # Print shap values dependencies ------------------------------------------
    # Loop over predictors
    for i, c_pred in enumerate(shap_explainations.feature_names):
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot SHAP Scatter plot
        scatter(shap_explainations[:, i],
                color='#777777',
                hist=True,
                axis_color='#333333',
                dot_size=16,
                x_jitter='auto',
                alpha=.5,
                title=None,
                xmin=None,
                xmax=None,
                ymin=None,
                ymax=None,
                overlay=None,
                ax=ax,
                ylabel='SHAP values',
                show=False)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel(ax.get_xlabel(), fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            task['ESTIMATOR_NAME']+' ' +
            'SHAP values for'+' ' +
            task['y_name'][0]+'\n' +
            'SHAP values = deviation from expected value (' +
            str(np.round(np.mean(
                [k.base_values for k in results['explainations']]),
                decimals=2)) +
            ')')
        # Add title
        plt.title(title_str, fontsize=10)

        # Save plot -------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     task['ESTIMATOR_NAME']+'_' +
                     'shap_values_dependency'+'_' +
                     task['y_name'][0]+'_' +
                     str(c_pred))[:135]
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()


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
    # Get SHAP effects --------------------------------------------------------
    # SHAP effects
    shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                    for k in results['explainations']]
    # Make dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])

    # Process SHAP effects-----------------------------------------------------
    # Mean shap values
    shap_effects_se_mean = shap_effects_df.mean(axis=0)
    # Sort from highto low
    shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
        ascending=False)

    # Get SHAP effects interactions -------------------------------------------
    # SHAP effects
    shap_effects_inter = np.array([np.mean(np.abs(k.values), axis=0)
                                   for k in results['explainations']])
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
    shap_effects_inter_sh = np.array([np.mean(np.abs(k.values), axis=0)
                                      for k in results['explainations_sh']])

    # Additional info ---------------------------------------------------------
    # x names lengths
    x_names_max_len = max([len(i) for i in task['x_names']])
    # x names count
    x_names_count = len(task['x_names'])

    # Make labels with pvales -------------------------------------------------
    # Get p values
    pval = np.zeros((shap_effects_inter.shape[1],
                     shap_effects_inter.shape[2]))
    # Iterate over shap_effects
    for x, y in np.ndindex((shap_effects_inter.shape[1],
                            shap_effects_inter.shape[2])):
        # Get current SHAP effect
        c_effect = shap_effects_inter[:, x, y]
        # Get current SHAP effect shuffle
        c_effect_sh = shap_effects_inter_sh[:, x, y]
        # Calculate p-value
        _, _, pval[x, y] = dep_two_sample_ttest(
            c_effect, c_effect_sh, task['TST_SIZE_FRAC'], side='one')
    # Multiple comparison correction
    if task['MCC']:
        # Multiply p value by number of tests
        pval = pval*(x_names_count**2)
        # Set p values > 1 to 1
        pval = pval.clip(None, 1)
    # Initialize labels dataframe
    interaction_labels_df = pd.DataFrame(np.zeros([
        shap_effects_inter.shape[1],
        shap_effects_inter.shape[2]]))
    # Iterate labels
    for x, y in np.ndindex((shap_effects_inter.shape[1],
                            shap_effects_inter.shape[2])):
        # Make label
        interaction_labels_df.iloc[x, y] = (
            str(np.around(shap_effects_inter_df.iloc[x, y],
                          decimals=2)) +
            '\n'+'p'+' ' +
            str(np.around(pval[x, y], decimals=3)))
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

    # Plot interaction effects ------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(figsize=(x_names_max_len*.1+x_names_count*1+1,
                                    x_names_max_len*.1+x_names_count*1+1))
    # Make colorbar string
    clb_str = ('mean(|SHAP value|)')
    # Plot confusion matrix
    sns.heatmap(shap_effects_inter_sort_df,
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
        task['ANALYSIS_NAME']+' ' +
        task['ESTIMATOR_NAME']+' ' +
        'SHAP effects for'+' ' +
        task['y_name'][0]+'\n' +
        'mean(|SHAP values|) = deviation from expected value (' +
        str(np.round(np.mean(
            [k.base_values for k in results['explainations']]),
            decimals=2)) +
        ')')
    # Add title
    plt.title(title_str, fontsize=10)
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar tick size
    cb_ax.tick_params(labelsize=10)
    # Modifying color bar fontsize
    cb_ax.set_ylabel(clb_str, fontsize=10)
    cb_ax.set_box_aspect(50)

    # Save plot ---------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 task['ESTIMATOR_NAME']+'_' +
                 'shap_effects_interactions'+'_' +
                 task['y_name'][0])[:135]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


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
    # Get SHAP explainations object -----------------------------------------
    shap_explainations = Explanation(np.vstack(
        [k.values for k in results['explainations']]),
        base_values=np.hstack(
            [k.base_values for k in results['explainations']]),
        data=np.vstack(
            [k.data for k in results['explainations']]),
        display_data=None,
        instance_names=None,
        feature_names=results['explainations'][0].feature_names,
        output_names=None,
        output_indexes=None,
        lower_bounds=None,
        upper_bounds=None,
        error_std=None,
        main_effects=None,
        hierarchical_values=None,
        clustering=None,
        compute_time=np.sum(
            [k.compute_time for k in results['explainations']]))

    # Print shap values dependencies ------------------------------------------
    # Make list of permutations
    permutations_list = (
        [(i, i) for i in shap_explainations.feature_names] +
        list(permutations(shap_explainations.feature_names, 2)))
    # Loop over predictor pairs
    for ind in permutations_list:
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot SHAP dependence
        dependence_plot(ind,
                        shap_values=shap_explainations.values,
                        features=pd.DataFrame(
                            shap_explainations.data,
                            columns=shap_explainations.feature_names),
                        feature_names=shap_explainations.feature_names,
                        display_features=None,
                        interaction_index='auto',
                        color='#1E88E5',
                        axis_color='#333333',
                        cmap=None,
                        dot_size=16,
                        x_jitter=0,
                        alpha=.66,
                        title=None,
                        xmin=None,
                        xmax=None,
                        ax=ax,
                        show=False,
                        ymin=None,
                        ymax=None)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            task['ESTIMATOR_NAME']+' ' +
            'SHAP interaction values for'+' ' +
            task['y_name'][0]+'\n' +
            'SHAP values = deviation from expected value (' +
            str(np.round(np.mean(
                [k.base_values for k in results['explainations']]),
                decimals=2)) +
            ')')
        # Add title
        ax.set_title(title_str, fontsize=10)
        # Check if mor than 1 axes are present
        if len(fig.axes) > 1:
            # Get colorbar
            cb_ax = fig.axes[1]
            # Modifying color bar tick size
            cb_ax.tick_params(labelsize=10)
            # Modifying color bar fontsize
            cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=10)

        # Save plot -------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     task['ESTIMATOR_NAME']+'_' +
                     'shap_interaction_values'+'_' +
                     task['y_name'][0]+'_' +
                     ind[0]+'_' +
                     ind[1])[:135]
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        plt.show()


def main():
    '''
    Main function of plot results of machine-learning based data analysis.

    Returns
    -------
    None.

    '''
    ###########################################################################
    # Specify analysis task
    ###########################################################################

    # Print hyper parameter distributions
    PPD = True
    # Do multiple comparison correction?
    MCC = False
    # Save plots additionally as svg?
    AS_SVG = False

    ###########################################################################

    # Get analysis results subdirs of current directory -----------------------
    res_dirs = [f.name for f in os.scandir('.')
                if f.is_dir() and (f.name.startswith('res_reg_')
                                   or f.name.startswith('res_clf_'))]

    # Loop over results dirs --------------------------------------------------
    # Loop over result dirs
    for res_dir in res_dirs:

        # Loop over tasks -----------------------------------------------------
        # Get task paths of current results subdir
        task_paths = [f.name for f in os.scandir('./'+str(res_dir)+'/')
                      if f.name.endswith('_task.pickle')]
        # Get results paths of current results subdir
        results_paths = [f.name for f in os.scandir('./'+str(res_dir)+'/')
                         if f.name.endswith('_results.pickle')]
        # Loop over tasks
        for i_task, task_path in enumerate(task_paths):

            # Load task and results -------------------------------------------
            # Load task description
            task = lfp(res_dir+'/'+task_path)
            # Add multiple comparison correction to task
            task['MCC'] = MCC
            # Add as svg to task
            task['AS_SVG'] = AS_SVG
            # Load results
            results = lfp(res_dir+'/'+results_paths[i_task])

            # Create plots directory ------------------------------------------
            # Plots path
            plots_path = res_dir+'/'+task['y_name'][0]+'_plots'
            # Create plots dir
            create_dir(plots_path)

            # Parameter distributions -----------------------------------------
            # Print parameter distributions
            if PPD:
                print_parameter_distributions(task, results, plots_path)

            # Model fit -------------------------------------------------------
            # Regressor
            if task['KIND'] == 'reg':
                # Print model fit as scatter plot
                print_regression_scatter(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_regression_violin(task, results, plots_path)
            # Classification
            elif task['KIND'] == 'clf':
                # Print model fit as confusion matrix
                print_classification_confusion(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_classification_violin(task, results, plots_path)
            # Other
            else:
                # Raise error
                raise TypeError('Kind not found.')

            # SHAP effects ----------------------------------------------------
            # Print SHAP effects
            print_shap_effects(task, results, plots_path)

            # SHAP effects distribution ---------------------------------------
            # Print SHAP effects distribution
            print_shap_effects_distribution(task, results, plots_path)

            # SHAP values -----------------------------------------------------
            # Print SHAP values
            print_shap_values(task, results, plots_path)

            # SHAP dependencies -----------------------------------------------
            # Print SHAP dependences
            print_shap_dependences(task, results, plots_path)

            # SHAP effects interactions ---------------------------------------
            # Print SHAP effects interactions
            if task['SHAP_INTERACTIONS']:
                print_shap_effects_interactions(task, results, plots_path)

            # SHAP interaction values -----------------------------------------
            # Print SHAP interaction values
            if task['SHAP_INTERACTIONS']:
                print_shap_interaction_values(task, results, plots_path)


if __name__ == '__main__':
    main()
