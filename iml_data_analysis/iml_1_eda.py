# *- coding: utf-8 -*-
'''
Interpretable Machine-Learning - Exploratory Data Analysis (EDA)
v185
@author: Dr. David Steyrl david.steyrl@univie.ac.at
'''

import math as mth
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import warnings
from itertools import permutations
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn_repeated_group_k_fold import RepeatedGroupKFold


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


def prepare(objective: str, num_classes: int) -> tuple:
    '''
    Prepare estimator, prepare seach_space.

    Parameters
    ----------
    objective : string
        String with objective describtion variables.
    num_classes : int
        Integer counting the number of classes

    Returns
    -------
    estimator : scikit-learn compatible estimator
        Prepared estimator object.
    space : dict
        Space that should be searched for optimale parameters.
    '''

    # Make estimator ----------------------------------------------------------
    # Regression
    if objective == 'regression':
        # Estimator
        estimator = LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=100,
            subsample_for_bin=100000,
            objective='huber',
            min_split_gain=0,
            min_child_weight=0,
            min_child_samples=10,
            subsample=1,
            subsample_freq=0,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=0,
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
    # Classification
    elif objective == 'classification':
        # Estimator
        estimator = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=100,
            subsample_for_bin=100000,
            objective='multiclass',
            class_weight='balanced',
            min_split_gain=0,
            min_child_weight=0,
            min_child_samples=10,
            subsample=1,
            subsample_freq=0,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=0,
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
               'num_class': num_classes,
               'use_quantized_grad': True,
               'verbosity': -1,
               })
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Make search space -------------------------------------------------------
    # Search space
    space = {
        'estimator__colsample_bytree': uniform(0.1, 0.9),
        'estimator__extra_trees': [True, False],
        'estimator__reg_lambda': loguniform(0.1, 100),
        }

    # Return estimator and space ----------------------------------------------
    return estimator, space


def split_data(df: pd.DataFrame, i_trn: np.ndarray,
               i_tst: np.ndarray) -> tuple:
    '''
    Split dataframe in training and testing dataframes.

    Parameters
    ----------
    df : dataframe
        Dataframe holding the data to split.
    i_trn : numpy ndarray
        Array with indices of training data.
    i_tst : numpy ndarray
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


def compute_pair_pred(task: dict, g: pd.Series, x: pd.Series, y: pd.Series,
                      objective: str) -> float:
    '''
    Compute pairwise prediction score (R², acc) of x and y.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : series
        Series holding the group data.
    x : series
        Series holding the predictor data.
    y : series
        Series holding the target data.
    objective : string
        String with objective describtion variables.

    Returns
    -------
    pair_pred : float
        Pairwise predictions scores (0-1). R² for regression,
        adjusted balanced accuracy for classification.
    '''

    # Initialize --------------------------------------------------------------
    # Initialize score list
    scores = []
    # Get number of classes if task is classification
    # Classification
    if objective == 'classification':
        # Number of unique classes in prediction target
        task['n_classes'] = y.nunique()[task['y_name']]
    # Regression
    elif objective == 'regression':
        # Set number of classes to 0 for compatibility
        task['n_classes'] = 0
    # Get estimator and space
    estimator, space = prepare(objective, task['n_classes'])

    # Main cross-validation loop ----------------------------------------------
    # Calculate number of repetition for outer CV
    task['n_rep_outer_cv'] = mth.ceil(task['N_PRED_OUTER_CV']/g.shape[0])
    # Instatiate main cv splitter with fixed random state for comparison
    cv = RepeatedGroupKFold(
        n_splits=task['N_CV_FOLDS'],
        n_repeats=task['n_rep_outer_cv'],
        random_state=None)
    # Loop over main (outer) cross validation splits
    for i_cv, (i_trn, i_tst) in enumerate(cv.split(g, groups=g)):

        # Split data ----------------------------------------------------------
        # Split groups
        g_trn, g_tst = split_data(g, i_trn, i_tst)
        # Split targets
        y_trn, y_tst = split_data(y, i_trn, i_tst)
        # Split predictors
        x_trn, x_tst = split_data(x, i_trn, i_tst)

        # Get scorer ----------------------------------------------------------
        # Regression
        if objective == 'regression':
            # R² score
            scorer = 'r2'
        # Classification
        elif objective == 'classification':
            # Balanced accuracy for classification
            scorer = 'balanced_accuracy'
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')

        # Tune analysis pipeline ----------------------------------------------
        # Choose n_repeats to approx N_SAMPLES_INNER_CV predictions
        task['n_rep_inner_cv'] = mth.ceil(
            task['N_PRED_INNER_CV'] / g_trn.shape[0])
        # Instatiate random parameter search
        search = RandomizedSearchCV(
            estimator,
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

        # Predict -------------------------------------------------------------
        # Predict test samples
        y_pred = search.best_estimator_.predict(x_tst)

        # Score results -------------------------------------------------------
        # Regression
        if objective == 'regression':
            # Score predictions in terms of R²
            scores.append(r2_score(y_tst, y_pred))
        # Classification
        elif objective == 'classification':
            # Calculate model fit in terms of acc
            scores.append(balanced_accuracy_score(
                y_tst, y_pred, adjusted=True))
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')

    # Process scores ----------------------------------------------------------
    # Limit pairwise predictions scores to be bigger than or equal to 0
    pair_pred = max(0, np.mean(scores))

    # Return pairwise predictions ---------------------------------------------
    return pair_pred


def eda(task: dict, g: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> None:
    '''
    Carries out exploratory data analysis, incl.:
    Data distribuations (1D, violinplot),
    Data distributions (2D, pairplots),
    Data joint information (linear, heatmap),
    Data joint information (non-linear, heatmap),
    Multidimensional pattern in data via PCA (linear, heatmap),
    Outlier in data (non-linear, histogram).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Dataframe holding the group data.
    x : dataframe
        Dataframe holding the predictor data.
    y : dataframe
        Dataframe holding the target data.

    Returns
    -------
    None.
    '''

    # Preprocessing -----------------------------------------------------------
    # Instatiate target encoder
    te = TargetEncoder(
        categories='auto',
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
         ('target', 'passthrough', task['y_name']),
         ],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Pipeline
    pre_pipe = Pipeline(
        [('coltrans', coltrans),
         ('std_scaler', StandardScaler())],
        memory=None,
        verbose=False).set_output(transform='pandas')
    # Concatinate predictors and targets
    z = pd.concat([x, y], axis=1)
    # Do preprocessing
    z = pre_pipe.fit_transform(z, y.squeeze())

    # 1D data distributions ---------------------------------------------------
    # Do 1D data distribution violin plot?
    if task['DATA_DISTRIBUTION_1D']:
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])
        # Create a figure
        fig, ax = plt.subplots(
            figsize=(x_names_max_len*.1+4, x_names_count*.7+1))
        # Violinplot all data
        sns.violinplot(
            data=z,
            bw_method='scott',
            bw_adjust=0.5,
            cut=2,
            density_norm='width',
            gridsize=100,
            width=0.8,
            inner='box',
            orient='h',
            linewidth=1,
            color='#777777',
            saturation=1.0,
            ax=ax)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set x ticks and size
        ax.set_xlabel('standardized range', fontsize=10)
        # Set y ticks and size
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
        # Add horizontal grid
        fig.axes[0].set_axisbelow(True)
        # Set grid style
        fig.axes[0].grid(
            axis='y',
            color='#bbbbbb',
            linestyle='dotted',
            alpha=.3)
        # Make title string
        title_str = (task['ANALYSIS_NAME']+'\n' +
                     'Data distributions (1D)\n')
        # set title
        plt.title(title_str, fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
            '_'+task['y_name'][0]+'_eda_1_distri_1D')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # 2D data distribution ----------------------------------------------------
    # Do 2D data distribution pairplot?
    if task['DATA_DISTRIBUTION_2D']:
        # Make pairplot
        pair_plot = sns.pairplot(
            z,
            corner=False,
            diag_kind='kde',
            plot_kws={'color': '#777777'},
            diag_kws={'color': '#777777'})
        # Make title string
        title_str = (task['ANALYSIS_NAME']+'\n' +
                     'Data distributions (2D)\n')
        # set title
        pair_plot.fig.suptitle(title_str, fontsize=10, y=1.0)
        # Add variable kde to plot
        pair_plot.map_lower(sns.kdeplot, levels=3, color='.2')

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
            '_'+task['y_name'][0]+'_eda_2_distri_2D')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # Joint information linear ------------------------------------------------
    # Do linear joint information via correlation heatmap plot?
    if task['DATA_JOINT_INFORMATION_LINEAR']:
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])
        # Create a figure
        fig, ax = plt.subplots(
            figsize=(x_names_count*.6+x_names_max_len*.1+1,
                     x_names_count*.6+x_names_max_len*.1+1))
        # Make colorbar string
        clb_str = ('correlation (-1 to 1)')
        # Print correlations
        sns.heatmap(
            z.corr(),
            vmin=-1,
            vmax=1,
            cmap='Greys',
            center=None,
            robust=True,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 10},
            linewidths=.5,
            linecolor='#999999',
            cbar=True,
            cbar_kws={'label': clb_str, 'shrink': 0.6},
            cbar_ax=None,
            square=True,
            xticklabels=1,
            yticklabels=1,
            mask=None,
            ax=ax)
        # This sets the yticks 'upright' with 0, as opposed to sideways with 90
        plt.yticks(rotation=0)
        # This sets the xticks 'sideways' with 90
        plt.xticks(rotation=90)
        # Make title string
        title_str = (task['ANALYSIS_NAME']+'\n' +
                     'Joint information in data (linear, correlation)\n')
        # set title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel(clb_str, fontsize=10)
        cb_ax.set_box_aspect(50)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
            '_'+task['y_name'][0]+'_eda_3_joint_lin'
            )
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # Joint information non linear --------------------------------------------
    # Do non-linear joint information via pairwise prediction heatmap plot?
    if task['DATA_JOINT_INFORMATION_NON_LINEAR']:
        # Check for NaN values
        if not z.isnull().values.any():
            # Make pairwise prediction matrix
            pair_pred = np.ones((len(list(z.columns)), len(list(z.columns))))
            # Make pairs
            for (id_pred1, id_pred2) in permutations(pd.factorize(
                    pd.Series(z.columns))[0], 2):
                # Make a mapping list between number and name
                mapping = list(z.columns)
                # Select task continous prediction target
                if mapping[id_pred2] in task['X_CON_NAMES']:
                    # Select objective
                    objective = 'regression'
                    # Get predictor data
                    xt = pd.DataFrame(z[mapping[id_pred1]])
                    # Get target data
                    yt = pd.DataFrame(z[mapping[id_pred2]])
                # Select task binary prediction target
                elif mapping[id_pred2] in task['X_CAT_BIN_NAMES']:
                    # Select objective
                    objective = 'classification'
                    # Get predictor data
                    xt = pd.DataFrame(z[mapping[id_pred1]])
                    # Get target data
                    yt = pd.DataFrame(pd.factorize(z[mapping[id_pred2]])[0],
                                      columns=[mapping[id_pred2]])
                # Select task multi class prediction target trat as regression
                elif mapping[id_pred2] in task['X_CAT_MULT_NAMES']:
                    # Select objective
                    objective = 'regression'
                    # Get predictor data
                    xt = pd.DataFrame(z[mapping[id_pred1]])
                    # Get target data
                    yt = pd.DataFrame(z[mapping[id_pred2]])
                # Select task target objective
                elif mapping[id_pred2] in task['Y_NAMES']:
                    # Select objective
                    objective = task['OBJECTIVE']
                    # Get predictor data
                    xt = pd.DataFrame(z[mapping[id_pred1]])
                    # Get target data select by objective regression
                    if objective == 'regression':
                        # Get target data
                        yt = pd.DataFrame(z[mapping[id_pred2]])
                    # Get target data select by objective other than regression
                    else:
                        # Get target data
                        yt = pd.DataFrame(
                            pd.factorize(z[mapping[id_pred2]])[0],
                            columns=[mapping[id_pred2]]
                            )
                # Other target objective
                else:
                    # Raise error
                    raise ValueError('OBJECTIVE not found.')
                # Compute pairwise prediction of current pair
                pair_pred[id_pred1, id_pred2] = compute_pair_pred(
                    task=task, g=g, x=xt, y=yt, objective=objective)
            # Names lengths
            names_max_len = max([len(i) for i in list(z.columns)])
            # Names count
            names_count = len(list(z.columns))
            # Create a figure
            fig, ax = plt.subplots(
                figsize=(names_count*.6+names_max_len*.1+1,
                         names_count*.6+names_max_len*.1+1))
            # Make colorbar string
            clb_str = ('joint information (0 to 1)')
            # Print pairwise predictions
            sns.heatmap(
                pair_pred,
                vmin=0,
                vmax=1,
                cmap='Greys',
                center=None,
                robust=True,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 10},
                linewidths=.5,
                linecolor='#999999',
                cbar=True,
                cbar_kws={'label': clb_str, 'shrink': 0.6},
                cbar_ax=None,
                square=True,
                xticklabels=list(z.columns),
                yticklabels=list(z.columns),
                mask=None,
                ax=ax)
            # Make title string
            title_str = (task['ANALYSIS_NAME']+'\n' +
                         'Joint information in data ' +
                         '(non-linear, pairwise predictions)\n' +
                         'y-axis: predictors, x-axis: prediction targets\n')
            # set title
            plt.title(title_str, fontsize=10)

            # Save figure -----------------------------------------------------
            # Make save path
            save_path = (
                task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                '_'+task['y_name'][0]+'_eda_4_joint_nonlin')
            # Save figure in .png format
            plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure in .svg format
                plt.savefig(save_path+'.svg', bbox_inches='tight')
            # show figure
            plt.show()
        # If nans
        else:
            # Raise warning
            warnings.warn(
                'Pairwise predictions skipped because of NaN values.'
                )

    # Multidimensional pattern with PCA ---------------------------------------
    # Do multidimensional pattern heatmap?
    if task['DATA_MULTIDIM_PATTERN']:
        # Check for NaN values
        if not z.isnull().values.any():
            # Instanciate PCA
            pca = PCA(
                n_components=x.shape[1],
                copy=True,
                whiten=False,
                svd_solver='auto',
                tol=0.0001,
                iterated_power='auto',
                random_state=None)
            # Fit PCA
            pca.fit(x)
            # x names count
            x_names_count = len(task['x_names'])
            # Make figure
            fig, ax = plt.subplots(figsize=(min((1+x_names_count*.6), 16), 4))
            # Plot data
            ax.plot(
                pca.explained_variance_ratio_,
                label='Explained variance per component')
            # Add dots
            ax.plot(
                pca.explained_variance_ratio_,
                color='black',
                marker='.',
                linestyle='None')
            # Set x limit
            ax.set_xlim((-0.01, ax.get_xlim()[1]))
            # Set y limit
            ax.set_ylim((-0.01, 1.01))
            # Remove top, right and left frame elements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Add x label
            ax.set_xlabel('PCA-component')
            # Add y label
            ax.set_ylabel('Explained Variance')
            # Create twin x axis
            ax2 = ax.twinx()
            # Plot cum sum of explained variance
            ax2.plot(
                np.cumsum(pca.explained_variance_ratio_),
                color='orange',
                label='Cumulative explained variance')
            # Add dots
            ax2.plot(
                np.cumsum(pca.explained_variance_ratio_),
                color='black',
                marker='.',
                linestyle='None')
            # Set x limit
            ax2.set_xlim((-0.01, ax2.get_xlim()[1]))
            # Set y limit
            ax2.set_ylim((-0.01, 1.01))
            # Remove top, right and left frame elements
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            # Add y label
            ax2.set_ylabel('Cumulative Variance')
            # Add labels to ax
            for comp, t in enumerate(
                    pca.explained_variance_ratio_.round(decimals=2)):
                # Add current label
                ax.text(comp, t, t, fontsize=10)
            # Add cum sum labels
            for comp, t in enumerate(
                    np.cumsum(
                        pca.explained_variance_ratio_).round(decimals=2)):
                # Add current cumsum label
                ax2.text(comp, t, t, fontsize=10)
            # Add legend
            fig.legend(
                loc='center right',
                bbox_to_anchor=(1, 0.5),
                bbox_transform=ax.transAxes)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+'\n' +
                'Multidimensinal pattern in data via PCA (linear)\n')
            # set title
            plt.title(title_str, fontsize=10)

            # Save figure -----------------------------------------------------
            # Make save path
            save_path = (
                task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                '_'+task['y_name'][0]+'_eda_5_pca')
            # Save figure in .png format
            plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure in .svg format
                plt.savefig(save_path+'.svg', bbox_inches='tight')
            # show figure
            plt.show()
        # If nans
        else:
            # Raise warning
            warnings.warn('PCA skipped because of NaN values.')

    # Outlier dection via Isolation Forests -----------------------------------
    # Do outlier detection histogram?
    if task['DATA_OUTLIER']:
        # Check for NaN values
        if not z.isnull().values.any():
            # Instanciate isolation forest
            iForest = IsolationForest(
                n_estimators=10000,
                max_samples='auto',
                contamination='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=-2,
                random_state=None,
                verbose=0,
                warm_start=False)
            # Fit data and predict outlier
            outlier = iForest.fit_predict(x)
            # Make outlier dataframe
            outlier_df = pd.DataFrame(data=outlier, columns=['is_outlier'])
            # Outlier score
            outlier_score = iForest.decision_function(x)
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot hist of inlier score
            sns.histplot(
                data=outlier_score,
                bins=30,
                kde=True,
                color='#777777',
                ax=ax)
            # Remove top, right and left frame elements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Add x label
            ax.set_xlabel('Isolation Forest outlier score')
            # Add y label
            ax.set_ylabel('Count')
            # Create title string
            title_str = (
                task['ANALYSIS_NAME']+'\n' +
                'Outlier in data via Isolation Forest: ' +
                '{:.1f}% potential outliers\n')
            # Add title
            ax.set_title(
                title_str.format(np.sum(outlier == -1)/len(outlier)*100))

            # Save figure -----------------------------------------------------
            # Make save path
            save_path = (
                task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                '_'+task['y_name'][0]+'_eda_6_iForest')
            # Save outlier data
            outlier_df.to_excel(save_path+'.xlsx')
            # Save figure in .png format
            plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure in .svg format
                plt.savefig(save_path+'.svg', bbox_inches='tight')
            # show figure
            plt.show()
        # If nans
        else:
            # Raise warning
            warnings.warn('Warning: Outlier skipped because of NaN values.')

    # Return ------------------------------------------------------------------
    return


def main() -> None:
    '''
    Main function of exploratory data analysis.

    Returns
    -------
    None.
    '''

    ###########################################################################
    # Specify analysis task
    ###########################################################################

    # 1. Specify task ---------------------------------------------------------

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
    # Number of predictions in outer CV. int (default: 1000)
    N_PRED_OUTER_CV = 1000
    # Number of tries in random search. int (default: 100)
    N_SAMPLES_RS = 100
    # Number of predictions in inner CV. int (default: 1000)
    N_PRED_INNER_CV = 1000
    # Save plots additionally AS_SVG? bool (default: False)
    AS_SVG = False

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
        'education',
        'environment_satisfaction',
        'job_satisfaction',
        'monthly_income',
        'num_companies_worked',
        'performance_rating',
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
    #     'proline'
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

    ###########################################################################

    # Create results directory path -------------------------------------------
    path_to_results = 'res_eda_'+ANALYSIS_NAME

    # Create task variable ----------------------------------------------------
    task = {
        'MAX_SAMPLES': MAX_SAMPLES,
        'N_JOBS': N_JOBS,
        'N_CV_FOLDS': N_CV_FOLDS,
        'DATA_DISTRIBUTION_1D': DATA_DISTRIBUTION_1D,
        'DATA_DISTRIBUTION_2D': DATA_DISTRIBUTION_2D,
        'DATA_JOINT_INFORMATION_LINEAR': DATA_JOINT_INFORMATION_LINEAR,
        'DATA_JOINT_INFORMATION_NON_LINEAR': DATA_JOINT_INFORMATION_NON_LINEAR,
        'DATA_MULTIDIM_PATTERN': DATA_MULTIDIM_PATTERN,
        'N_PRED_OUTER_CV': N_PRED_OUTER_CV,
        'N_PRED_INNER_CV': N_PRED_INNER_CV,
        'N_SAMPLES_RS': N_SAMPLES_RS,
        'DATA_OUTLIER': DATA_OUTLIER,
        'AS_SVG': AS_SVG,
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
        'path_to_results': path_to_results,
        'x_names': X_CON_NAMES+X_CAT_BIN_NAMES+X_CAT_MULT_NAMES,
        }

    # Create results directory ------------------------------------------------
    create_dir(path_to_results)

    # Copy this python script to results directory ----------------------------
    shutil.copy('iml_1_eda.py', path_to_results+'/iml_1_eda.py')

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

    # Prepare data ------------------------------------------------------------
    # Iterate over prediction targets (Y_NAMES)
    for i_y, y_name in enumerate(Y_NAMES):
        # Add prediction target index to task
        task['i_y'] = i_y
        # Add prediction target name to task
        task['y_name'] = [y_name]

        # Deal with NaNs in the target ----------------------------------------
        # Get current target and remove NaNs
        y = Y[y_name].to_frame().dropna()
        # Use y index for groups and reset index
        g = G.reindex(index=y.index).reset_index(drop=True)
        # Use y index for predictors and reset index
        x = X.reindex(index=y.index).reset_index(drop=True)
        # Reset index of target
        y = y.reset_index(drop=True)

        # Limit number of samples ---------------------------------------------
        # Subsample predictors
        x = x.sample(
            n=min(x.shape[0], task['MAX_SAMPLES']),
            random_state=None,
            ignore_index=False)
        # Slice group to fit subsampled predictors
        g = g.loc[x.index, :].reset_index(drop=True)
        # Slice targets to fit subsampled predictors
        y = y.loc[x.index, :].reset_index(drop=True)
        # Reset index of predictors
        x = x.reset_index(drop=True)

        # Store data ----------------------------------------------------------
        # Save groups
        g.to_excel(
            path_to_results+'/'+ANALYSIS_NAME+'_'+task['y_name'][0] +
            '_data_g.xlsx')
        # Save predictors
        x.to_excel(
            path_to_results+'/'+ANALYSIS_NAME+'_'+task['y_name'][0] +
            '_data_x.xlsx')
        # Save targets
        y.to_excel(
            path_to_results+'/'+ANALYSIS_NAME+'_'+task['y_name'][0] +
            '_data_y.xlsx')

        # Exploratory data analysis (EDA) -------------------------------------
        # Run EDA
        eda(task, g, x, y)

    # Return ------------------------------------------------------------------
    return


if __name__ == '__main__':
    main()
