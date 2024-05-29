# -*- coding: utf-8 -*-
'''
Interpretable Machine-Learning - Fairness (FRS)
v029
@author: Dr. David Steyrl david.steyrl@univie.ac.at
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import warnings
from scipy.stats import t
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score


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
    Computes two-tailed paired t-test with corrected variance.
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
    t_stat = abs(mean / std)
    # Compute p value for one-tailed t-test
    p_val = t.sf(t_stat, df=len(differences)-1)

    # Return t statistics and p value -----------------------------------------
    return t_stat, p_val


def main():
    '''
    Main function of plot results of machine-learning based data analysis.

    Returns
    -------
    None.
    '''

    ###########################################################################
    # Specify fairness checks
    ###########################################################################

    # Specify variable names to check fairness for. list of strings
    FAIR_NAMES = [
        'age',
        'sex',
        'sepal_length',
        ]
    # Specify thresholds to check fairness for. list of strings
    FAIR_THRESHOLDS = [
        50,
        1.5,
        6,
        ]
    # Save plots additionally as svg. bool (default: False)
    AS_SVG = False

    ###########################################################################

    # Load result paths -------------------------------------------------------
    res_paths = [f.name for f in os.scandir('.')
                 if f.is_dir() and f.name.startswith('res_iml_')]

    # Loop over result paths --------------------------------------------------
    for res_path in res_paths:
        # Get group paths
        g_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                   if f.name.endswith('_g.xlsx')]
        # Get predictor paths
        x_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                   if f.name.endswith('_x.xlsx')]
        # Get target paths
        y_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                   if f.name.endswith('_y.xlsx')]
        # Get task paths
        task_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                      if f.name.endswith('_task.pickle')]
        # Get result paths
        results_paths = [f.name for f in os.scandir('./'+str(res_path)+'/')
                         if f.name.endswith('_results.pickle')]

        # Loop over tasks -----------------------------------------------------
        for (g_path, x_path, y_path, task_path, results_path) in zip(
                g_paths, x_paths, y_paths, task_paths, results_paths):
            # Load predictor data
            x = pd.read_excel(
                res_path+'/'+x_path,
                sheet_name='Sheet1',
                header=0,
                dtype=np.float64)
            # Load task description
            task = lfp(res_path+'/'+task_path)
            # Add as svg to task
            task['AS_SVG'] = AS_SVG
            # Load results
            results = lfp(res_path+'/'+results_path)
            # Plots path
            plots_path = res_path+'/'+task['y_name']+'_fair'
            # Create plots dir
            create_dir(plots_path)

            # Loop over fainess checks ----------------------------------------
            for (fair_name, fair_treshold) in zip(FAIR_NAMES, FAIR_THRESHOLDS):

                # Get predictor -----------------------------------------------
                # Try to get fair predictor
                try:
                    # Get fair predictor
                    fair_predictor = x[fair_name]
                except KeyError:
                    # Issue a warning
                    warnings.warn(
                        fair_name +
                        ' not in predictors of ' +
                        task['ANALYSIS_NAME'])
                    # Continue for loop
                    continue

                # Get grouping index ------------------------------------------
                # Get fair decision
                fair_decision = fair_predictor >= fair_treshold
                # Get fair index of true
                fair_index_true = \
                    fair_predictor.index[fair_decision].to_numpy()
                # Get fair index of false
                fair_index_false = \
                    fair_predictor.index[~fair_decision].to_numpy()

                # Extract samples, get errors, compute performance measures ---
                # Initialise errors_true list
                errors_true = []
                # Initialise errors_false list
                errors_false = []
                # Initialise perf_true list
                perf_true = []
                # Initialise perf_false list
                perf_false = []
                # Loop over cv repetitions
                for score in results['scores']:
                    # Get y_pred where fair index is true
                    y_pred_true = score['y_pred'][np.flatnonzero(np.isin(
                        score['y_ind'],
                        fair_index_true))]
                    # Get y_true where fair index is true
                    y_true_true = score['y_true'][np.flatnonzero(np.isin(
                        score['y_ind'],
                        fair_index_true))]
                    # Get errors in true sample
                    errors_true.append(
                        np.mean(np.abs(y_true_true - y_pred_true)))
                    # Regression
                    if task['OBJECTIVE'] == 'regression':
                        # Get perf of fair index is true samples
                        perf_true.append(r2_score(y_true_true, y_pred_true))
                    # Classification
                    else:
                        # Get perf of fair index is true samples
                        perf_true.append(balanced_accuracy_score(
                            y_true_true,
                            y_pred_true))
                    # Get y_pred where fair index is false
                    y_pred_false = score['y_pred'][np.flatnonzero(np.isin(
                        score['y_ind'],
                        fair_index_false))]
                    # Get y_true where fair index is false
                    y_true_false = score['y_true'][np.flatnonzero(np.isin(
                        score['y_ind'],
                        fair_index_false))]
                    # Get errors in false sample
                    errors_false.append(
                        np.mean(np.abs(y_true_false - y_pred_false)))
                    # Regression
                    if task['OBJECTIVE'] == 'regression':
                        # Get perf of fair index is false samples
                        perf_false.append(r2_score(y_true_false, y_pred_false))
                    # Classification
                    else:
                        # Get perf of fair index is false samples
                        perf_false.append(balanced_accuracy_score(
                            y_true_false,
                            y_pred_false))

                # Histogram of predictor values -------------------------------
                # Make figure
                fig, ax = plt.subplots(figsize=(8, 5))
                # Plot hist of inlier score
                sns.histplot(
                    data=fair_predictor,
                    bins=30,
                    kde=True,
                    color='#777777',
                    log_scale=False,
                    ax=ax)
                # Remove top, right and left frame elements
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # Set title
                ax.set_title(
                    task['ANALYSIS_NAME']+'\n' +
                    'Distribution of predictor values over samples',
                    fontsize=10)

                # Save figure -------------------------------------------------
                # Make save path
                save_path = (
                    plots_path+'/' +
                    task['ANALYSIS_NAME']+'_' +
                    task['y_name']+'_' +
                    fair_name+'_' +
                    'values')[:150]
                # Save figure
                plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
                # Check if save as svg is enabled
                if task['AS_SVG']:
                    # Save figure
                    plt.savefig(save_path+'.svg',  bbox_inches='tight')
                # Show figure
                plt.show()

                # Histogram of mean absolute error of each CV fold ------------
                # Make figure
                fig, ax = plt.subplots(figsize=(8, 5))
                # Plot hist of inlier score
                sns.histplot(
                    data=pd.DataFrame(
                        data=zip(errors_true, errors_false),
                        columns=[fair_name+' \u2265 '+str(fair_treshold),
                                 fair_name+' < '+str(fair_treshold)]),
                    bins=30,
                    kde=True,
                    palette=['#777777', '#000000'],
                    log_scale=False,
                    ax=ax)
                # Remove top, right and left frame elements
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # Add x label
                ax.set_xlabel('Mean absolute error per cv fold')
                # Add y label
                ax.set_ylabel('Count')
                # Calculate p-value between errors_true and errors_false
                _, pval_errors = corrected_ttest(
                    np.array(errors_true)-np.array(errors_false))
                # Make pval string
                if pval_errors <= 0.001:
                    pval_string = 'p\u22640.001'
                else:
                    pval_string = 'p={:.3f}'.format(pval_errors)
                # Set title
                ax.set_title(
                    task['ANALYSIS_NAME']+'\n' +
                    'Distribution of mean absolute error over CV folds \n' +
                    'Mean absolute error of ' +
                    fair_name+' \u2265 '+str(fair_treshold) + ' and ' +
                    fair_name+' < '+str(fair_treshold)+': ' +
                    str(np.round(np.mean(errors_true), decimals=2)) +
                    ' and ' +
                    str(np.round(np.mean(errors_false), decimals=2))+'\n' +
                    'Difference in means: ' +
                    str(np.round(np.mean(errors_true)-np.mean(errors_false),
                                 decimals=2)) +
                    ' | '+pval_string,
                    fontsize=10)

                # Save figure -------------------------------------------------
                # Make save path
                save_path = (
                    plots_path+'/' +
                    task['ANALYSIS_NAME']+'_' +
                    task['y_name']+'_' +
                    fair_name+'_' +
                    'error')[:150]
                # Save figure
                plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
                # Check if save as svg is enabled
                if task['AS_SVG']:
                    # Save figure
                    plt.savefig(save_path+'.svg',  bbox_inches='tight')
                # Show figure
                plt.show()

                # Histogram of performance of each CV fold --------------------
                # Make figure
                fig, ax = plt.subplots(figsize=(8, 5))
                # Plot hist of inlier score
                sns.histplot(
                    data=pd.DataFrame(
                        data=zip(perf_true, perf_false),
                        columns=[fair_name+' \u2265 '+str(fair_treshold),
                                 fair_name+' < '+str(fair_treshold)]),
                    bins=30,
                    kde=True,
                    palette=['#777777', '#000000'],
                    log_scale=False,
                    ax=ax)
                # Remove top, right and left frame elements
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # Regression
                if task['OBJECTIVE'] == 'regression':
                    # Add x label
                    ax.set_xlabel('R² per CV fold')
                # Classification
                else:
                    # Add x label
                    ax.set_xlabel('Balanced accuracy per CV fold')
                # Add y label
                ax.set_ylabel('Count')
                # Calculate p-value between perf_true and perf_false
                _, pval_perf = \
                    corrected_ttest(np.array(perf_true)-np.array(perf_false))
                # Make pval string
                if pval_perf <= 0.001:
                    pval_string = 'p\u22640.001'
                else:
                    pval_string = 'p={:.3f}'.format(pval_perf)
                # Regression
                if task['OBJECTIVE'] == 'regression':
                    # Set title
                    ax.set_title(
                        task['ANALYSIS_NAME']+'\n' +
                        'Distribution of R² over CV folds \n' +
                        'Mean R² of ' +
                        fair_name+' \u2265 '+str(fair_treshold) + ' and ' +
                        fair_name+' < '+str(fair_treshold)+': ' +
                        str(np.round(np.mean(perf_true), decimals=2)) +
                        ' and ' +
                        str(np.round(np.mean(perf_false), decimals=2))+'\n' +
                        'Difference in means: ' +
                        str(np.round(np.mean(perf_true)-np.mean(perf_false),
                                     decimals=2)) +
                        ' | '+pval_string,
                        fontsize=10)
                # Classification
                else:
                    # Set title
                    ax.set_title(
                        task['ANALYSIS_NAME']+'\n' +
                        'Distribution of balanced accuracy over CV folds \n' +
                        'Mean accuracy of ' +
                        fair_name+' \u2265 '+str(fair_treshold) + ' and ' +
                        fair_name+' < '+str(fair_treshold)+': ' +
                        str(np.round(np.mean(perf_true), decimals=2)) +
                        ' and ' +
                        str(np.round(np.mean(perf_false), decimals=2))+'\n' +
                        'Difference in means: ' +
                        str(np.round(np.mean(perf_true)-np.mean(perf_false),
                                     decimals=2)) +
                        ' | '+pval_string,
                        fontsize=10)

                # Save figure -------------------------------------------------
                # Make save path
                save_path = (
                    plots_path+'/' +
                    task['ANALYSIS_NAME']+'_' +
                    task['y_name']+'_' +
                    fair_name+'_' +
                    'perf')[:150]
                # Save figure
                plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
                # Check if save as svg is enabled
                if task['AS_SVG']:
                    # Save figure
                    plt.savefig(save_path+'.svg',  bbox_inches='tight')
                # Show figure
                plt.show()
    return


if __name__ == '__main__':
    main()
