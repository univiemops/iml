# *- coding: utf-8 -*-
'''
Interpretable Machine-Learning - Exploratory Data Analysis (EDA)
v140
@author: Dr. David Steyrl david.steyrl@univie.ac.at
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings('ignore', 'The figure layout has changed to tight')


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


def eda(task, x, y):
    '''
    Carries out exploratory data analysis, incl.:
    distribuation (violinplot),
    pairplots,
    correlation (heatmap),
    PCA,
    outlier.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
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
        cv=5,
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

    # Distributions -----------------------------------------------------------
    # Do VIOLINPLOTS?
    if task['VIOLINPLOTS']:
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
            saturation=1,
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
        title_str = (task['ANALYSIS_NAME']+' '+'data distributions')
        # set title
        plt.title(title_str, fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
            '_'+task['y_name'][0]+'_eda_1_distribuations')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # Pairplots ---------------------------------------------------------------
    # Do PAIRPLOTS?
    if task['PAIRPLOTS']:
        # Make pairplot
        pair_plot = sns.pairplot(
            z,
            corner=False,
            diag_kind='kde',
            plot_kws={'color': '#777777'},
            diag_kws={'color': '#777777'})
        # Make title string
        title_str = (task['ANALYSIS_NAME']+' '+'data pair plots')
        # set title
        pair_plot.fig.suptitle(title_str, fontsize=10, y=1.0)
        # Add variable kde to plot
        pair_plot.map_lower(sns.kdeplot, levels=3, color='.2')

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (
            task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
            '_'+task['y_name'][0]+'_eda_2_pairplots')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # Correlation -------------------------------------------------------------
    # Do HEATMAP?
    if task['HEATMAP']:
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])
        # Create a figure
        fig, ax = plt.subplots(
            figsize=(x_names_count*.5+x_names_max_len*.1+1,
                     x_names_count*.5+x_names_max_len*.1+1))
        # Make colorbar string
        clb_str = ('correlation coefficient')
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
        title_str = (task['ANALYSIS_NAME']+' '+'data correlations')
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
            '_'+task['y_name'][0]+'_eda_3_correlations')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # PCA and linear dependency -----------------------------------------------
    # Do PCA?
    if task['PCA']:
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
            fig, ax = plt.subplots(figsize=(max((1+x_names_count*.25), 8), 4))
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
            # Remove top, right and left frame elements
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            # Add y label
            ax2.set_ylabel('Cumulative Variance')
            # Add labels to ax
            for comp, t in enumerate(
                    pca.explained_variance_ratio_.round(decimals=2)):
                # Add current label
                ax.text(comp, t, t, fontsize=8)
            # Add cum sum labels
            for comp, t in enumerate(
                    np.cumsum(
                        pca.explained_variance_ratio_).round(decimals=2)):
                # Add current cumsum label
                ax2.text(comp, t, t, fontsize=8)
            # Add legend
            fig.legend(
                loc='center right',
                bbox_to_anchor=(1, 0.5),
                bbox_transform=ax.transAxes)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'data principle components variance contribution')
            # set title
            plt.title(title_str, fontsize=10)

            # Save figure -----------------------------------------------------
            # Make save path
            save_path = (
                task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                '_'+task['y_name'][0]+'_eda_4_pca')
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

    # Outlier dection with Isolation Forests ----------------------------------
    # Do OUTLIER detection?
    if task['OUTLIER']:
        # Check for NaN values
        if not z.isnull().values.any():
            # Instanciate isolation forest
            iForest = IsolationForest(
                n_estimators=1000,
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
            ax.set_ylabel('Number of samples')
            # Create title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'data outlier detection | Isolation Forest | outlier {:.1f} %')
            # Add title
            ax.set_title(
                title_str.format(np.sum(outlier == -1)/len(outlier)*100))

            # Save figure -----------------------------------------------------
            # Make save path
            save_path = (
                task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                '_'+task['y_name'][0]+'_eda_5_outlier')
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


def main():
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
    # Specify max number of samples. (default: 1000)
    MAX_SAMPLES = 1000
    # Do VIOLINPLOTS in EDA?
    VIOLINPLOTS = True
    # Do PAIRPLOTS in EDA?
    PAIRPLOTS = True
    # Do correlation HEATMAP in EDA?
    HEATMAP = True
    # Do PCA in EDA?
    PCA = True
    # Use Isolation Forest to automatically detect OUTLIERs in EDA?
    OUTLIER = True
    # Save plots additionally AS_SVG?
    AS_SVG = False

    # 2. Specify data ---------------------------------------------------------

    # Cancer data - classification 2 class, unbalanced classes
    # Specifiy an analysis name
    ANALYSIS_NAME = 'cancer'
    # Specify path to data. string
    PATH_TO_DATA = 'data/cancer_20230927.xlsx'
    # Specify sheet name. string
    SHEET_NAME = 'data'
    # Specify continous predictor names. list of string or []
    X_CON_NAMES = [
        'mean_radius',
        'mean_texture',
        'mean_perimeter',
        'mean_area',
        'mean_smoothness',
        'mean_compactness',
        'mean_concavity',
        'mean_concave_points',
        'mean_symmetry',
        'mean_fractal_dimension',
        'radius_error',
        'texture_error',
        'perimeter_error',
        'area_error',
        'smoothness_error',
        'compactness_error',
        'concavity_error',
        'concave_points_error',
        'symmetry_error',
        'fractal_dimension_error',
        'worst_radius',
        'worst_texture',
        'worst_perimeter',
        'worst_area',
        'worst_smoothness',
        'worst_compactness',
        'worst_concavity',
        'worst_concave_points',
        'worst_symmetry',
        'worst_fractal_dimension',
        ]
    # Specify binary categorical predictor names. list of string or []
    X_CAT_BIN_NAMES = []
    # Specify multi categorical predictor names. list of string or []
    X_CAT_MULT_NAMES = []
    # Specify target name(s). list of strings or []
    Y_NAMES = [
        'target',
        ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []

    # # Diabetes data - regression, binary category predictor
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'diabetes'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/diabetes_20230809.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
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
    #     's6',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = [
    #     'sex',
    #     ]
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'progression',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Digits data - classification 10 class, multicategory predictors
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'digits'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/digit_20230809.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = []
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = [
    #     'pixel_0_1',
    #     'pixel_0_2',
    #     'pixel_0_3',
    #     'pixel_0_4',
    #     'pixel_0_5',
    #     'pixel_0_6',
    #     'pixel_0_7',
    #     'pixel_1_0',
    #     'pixel_1_1',
    #     'pixel_1_2',
    #     'pixel_1_3',
    #     'pixel_1_4',
    #     'pixel_1_5',
    #     'pixel_1_6',
    #     'pixel_1_7',
    #     'pixel_2_0',
    #     'pixel_2_1',
    #     'pixel_2_2',
    #     'pixel_2_3',
    #     'pixel_2_4',
    #     'pixel_2_5',
    #     'pixel_2_6',
    #     'pixel_2_7',
    #     'pixel_3_0',
    #     'pixel_3_1',
    #     'pixel_3_2',
    #     'pixel_3_3',
    #     'pixel_3_4',
    #     'pixel_3_5',
    #     'pixel_3_6',
    #     'pixel_3_7',
    #     'pixel_4_0',
    #     'pixel_4_1',
    #     'pixel_4_2',
    #     'pixel_4_3',
    #     'pixel_4_4',
    #     'pixel_4_5',
    #     'pixel_4_6',
    #     'pixel_4_7',
    #     'pixel_5_0',
    #     'pixel_5_1',
    #     'pixel_5_2',
    #     'pixel_5_3',
    #     'pixel_5_4',
    #     'pixel_5_5',
    #     'pixel_5_6',
    #     'pixel_5_7',
    #     'pixel_6_0',
    #     'pixel_6_1',
    #     'pixel_6_2',
    #     'pixel_6_3',
    #     'pixel_6_4',
    #     'pixel_6_5',
    #     'pixel_6_6',
    #     'pixel_6_7',
    #     'pixel_7_0',
    #     'pixel_7_1',
    #     'pixel_7_2',
    #     'pixel_7_3',
    #     'pixel_7_4',
    #     'pixel_7_5',
    #     'pixel_7_6',
    #     'pixel_7_7',
    #     ]
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'digit',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Housing data - regression, multicategory predictor
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'housing'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/housing_20230907.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
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

    # # Iris data - classification 2 class,
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'iris_2'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/iris_20230809.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data_2class'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'sepal_length',
    #     'sepal_width',
    #     'petal_length',
    #     'petal_width',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'type',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Iris data - classification 3 class,
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'iris_3'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/iris_20230809.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data_3class'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'sepal_length',
    #     'sepal_width',
    #     'petal_length',
    #     'petal_width',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = []
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'type',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Radon data - regression, binary and multicategory predictors
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'radon'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/radon_20230809.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'Uppm',
    #     ]
    # # Specify binary categorical predictor names. list of string or []
    # X_CAT_BIN_NAMES = [
    #     'basement',
    #     'floor',
    #     ]
    # # Specify multi categorical predictor names. list of string or []
    # X_CAT_MULT_NAMES = [
    #     'county_code',
    #     ]
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'log_radon',
    #     ]
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    ###########################################################################

    # Create results directory path -------------------------------------------
    path_to_results = 'res_eda_'+ANALYSIS_NAME

    # Create task variable ----------------------------------------------------
    task = {
        'MAX_SAMPLES': MAX_SAMPLES,
        'VIOLINPLOTS': VIOLINPLOTS,
        'PAIRPLOTS': PAIRPLOTS,
        'HEATMAP': HEATMAP,
        'PCA': PCA,
        'OUTLIER': OUTLIER,
        'AS_SVG': AS_SVG,
        'ANALYSIS_NAME': ANALYSIS_NAME,
        'PATH_TO_DATA': PATH_TO_DATA,
        'SHEET_NAME': SHEET_NAME,
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
    shutil.copy('ml_1_eda.py', path_to_results+'/ml_1_eda.py')

    # Load data ---------------------------------------------------------------
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
        # Use y index for predictors and reset index
        x = X.reindex(index=y.index).reset_index(drop=True)
        # Reset index of target
        y = y.reset_index(drop=True)

        # Limit number of samples ---------------------------------------------
        # Subsample predictors
        x = x.sample(
            n=min(x.shape[0], task['MAX_SAMPLES']),
            random_state=3141592,
            ignore_index=False)
        # Slice targets to fit subsampled predictors
        y = y.loc[x.index, :].reset_index(drop=True)
        # Reset index of predictors
        x = x.reset_index(drop=True)

        # Store data ----------------------------------------------------------
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
        eda(task, x, y)

    # Return ------------------------------------------------------------------
    return


if __name__ == '__main__':
    main()
