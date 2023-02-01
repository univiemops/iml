# -*- coding: utf-8 -*-
'''
Exploratory data analysis
v109
@author: Dr. David Steyrl david.steyrl@gmail.com
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import warnings
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    'ignore',
    'iteritems is deprecated and will be removed in a future version.')


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
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)


def drop_nan_rows(x, y):
    '''
    Identify and drop rows containing nans from dataframes.

    Parameters
    ----------
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Targets dataframe.

    Returns
    -------
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Targets dataframe.

    '''
    # Search for nans in predictors
    rows_nans = list(x.loc[x.isna().any(axis=1).to_numpy(), :].index)
    # Drop rows from predictors
    x = x.drop(rows_nans).reset_index(drop=True)
    # Drop rows from targets
    y = y.drop(rows_nans).reset_index(drop=True)
    # Return x and y
    return x, y


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
    # Prepare data ------------------------------------------------------------
    # Create all data dataframe
    z = pd.concat([x, y], axis=1)
    # Instanciate scaler
    scaler = StandardScaler(copy=True,
                            with_mean=True,
                            with_std=True).set_output(transform='pandas')
    # Fit transform all data
    z_sc = scaler.fit_transform(z)
    # Fit transform predictors
    x_sc = scaler.fit_transform(x)

    # Distributions -----------------------------------------------------------
    # Do VIOLINPLOTS?
    if task['VIOLINPLOTS']:
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])
        # Create a figure
        fig, ax = plt.subplots(figsize=(x_names_max_len*.1+6,
                                        x_names_count*.4+1))
        # Violinplot all data
        sns.violinplot(data=z_sc, bw='scott', cut=2, scale='width',
                       gridsize=100, width=0.8, inner='box', orient='h',
                       linewidth=1, color='#777777', saturation=1, ax=ax)
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
        fig.axes[0].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'data distributions')
        # set title
        plt.title(title_str, fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                     '_eda_1_distribuations')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
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
        pair_plot = sns.pairplot(z_sc, corner=False, diag_kind='kde',
                                 plot_kws={'color': '#777777'},
                                 diag_kws={'color': '#777777'})
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'data pair plots')
        # set title
        pair_plot.fig.suptitle(title_str, fontsize=10, y=1.0)
        # Add variable kde to plot
        pair_plot.map_lower(sns.kdeplot, levels=3, color='.2')

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                     '_eda_2_pairplots')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
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
        fig, ax = plt.subplots(figsize=(x_names_count*.5+x_names_max_len*.1+1,
                                        x_names_count*.5+x_names_max_len*.1+1))
        # Make colorbar string
        clb_str = ('correlation coefficient')
        # Print correlations
        sns.heatmap(z_sc.corr(),
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
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'data correlations')
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
        save_path = (task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                     '_eda_3_correlations')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show plot
        plt.show()

    # PCA and linear dependency -----------------------------------------------
    # Do PCA?
    if task['PCA']:
        # Instanciate PCA
        pca = PCA(n_components=x.shape[1],
                  copy=True,
                  whiten=False,
                  svd_solver='auto',
                  tol=1e-4,
                  iterated_power='auto',
                  random_state=None)
        # Fit PCA
        pca.fit(x_sc)
        # x names count
        x_names_count = len(task['x_names'])
        # Make figure
        fig, ax = plt.subplots(figsize=(max((1+x_names_count*.25), 8),
                                        4))
        # Plot data
        ax.plot(pca.explained_variance_ratio_,
                label='Explained variance per component')
        # Plot cum sum of explained variance
        ax.plot(np.cumsum(pca.explained_variance_ratio_),
                label='Cumulative explained variance')
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add x label
        ax.set_xlabel('PCA-component')
        # Add y label
        ax.set_ylabel('Relative Variance')
        # Add legend
        ax.legend()
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'data principle components variance contribution')
        # set title
        plt.title(title_str, fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                     '_eda_4_pca')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # show figure
        plt.show()

    # Outlier dection with Isolation Forests ----------------------------------
    # Do OUTLIER detection?
    if task['OUTLIER']:
        # Instanciate isolation forest
        iForest = IsolationForest(n_estimators=5000,
                                  max_samples='auto',
                                  contamination='auto',
                                  max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=-2,
                                  random_state=None,
                                  verbose=0,
                                  warm_start=False)
        # Fit data and predict outlier
        outlier = iForest.fit_predict(x_sc)
        # Make outlier dataframe
        outlier_df = pd.DataFrame(data=outlier, columns=['is_outlier'])
        # Outlier score
        outlier_score = iForest.decision_function(x_sc)
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(data=outlier_score,
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
        ax.set_title(title_str.format(np.sum(outlier == -1)/len(outlier)*100))

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (task['path_to_results']+'/'+task['ANALYSIS_NAME'] +
                     '_eda_5_outlier')
        # Save outlier data
        outlier_df.to_excel(save_path+'.xlsx')
        # Save figure in .png format
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure in .svg format
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # show figure
        plt.show()


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
    # Specify max number of samples. Default: 1000.
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
    # Drop rows with nans. If false imputation & ohe of nans (default: False)
    DROP_NAN = False

    # 2. Specify data ---------------------------------------------------------

    # Cancer data - classification, 2 classes
    # Specifiy an analysis name
    ANALYSIS_NAME = 'cancer'
    # Specify path to data. string
    PATH_TO_DATA = 'data/cancer_20221123.xlsx'
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
        'worst_fractal_dimension']
    # Specify categorical predictor names. list of string or []
    X_CAT_NAMES = []
    # Specify target name(s). list of strings or []
    Y_NAMES = [
        'target']
    # Rows to skip. list of int or []
    SKIP_ROWS = []

    # # Diabetes data - regression
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'diabetes'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/diabetes_20220824.xlsx'
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
    #     's6']
    # # Specify categorical predictor names. list of string or []
    # X_CAT_NAMES = [
    #     'sex']
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'target']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Housing data - regression, 20k samples, categorical predictor
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'housing'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/housing_20220824.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'longitude',
    #     'latitude',
    #     'housing_median_age',
    #     'total_rooms',
    #     'total_bedrooms',
    #     'population',
    #     'households',
    #     'median_income']
    # # Specify categorical predictors names. list of string or []
    # X_CAT_NAMES = [
    #     'ocean_proximity']
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'median_house_value_k']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Radon data - regression, categorical predictors with high cardiality
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'radon'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/radon_20220824.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
    # # Specify continous predictor names. list of string or []
    # X_CON_NAMES = [
    #     'Uppm']
    # # Specify categorical predictors names. list of string or []
    # X_CAT_NAMES = [
    #     'county_code',
    #     'floor']
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'log_radon']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    # # Wine data - classification, 3 classes
    # # Specifiy an analysis name
    # ANALYSIS_NAME = 'wine'
    # # Specify path to data. string
    # PATH_TO_DATA = 'data/wine_20221122.xlsx'
    # # Specify sheet name. string
    # SHEET_NAME = 'data'
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
    #     'proline']
    # # Specify categorical predictor names. list of string or []
    # X_CAT_NAMES = []
    # # Specify target name(s). list of strings or []
    # Y_NAMES = [
    #     'target']
    # # Rows to skip. list of int or []
    # SKIP_ROWS = []

    ###########################################################################

    # Create results directory path -------------------------------------------
    path_to_results = 'res_eda_'+ANALYSIS_NAME

    # Create task variable ----------------------------------------------------
    task = {'MAX_SAMPLES': MAX_SAMPLES,
            'VIOLINPLOTS': VIOLINPLOTS,
            'PAIRPLOTS': PAIRPLOTS,
            'HEATMAP': HEATMAP,
            'PCA': PCA,
            'OUTLIER': OUTLIER,
            'AS_SVG': AS_SVG,
            'DROP_NAN': DROP_NAN,
            'ANALYSIS_NAME': ANALYSIS_NAME,
            'PATH_TO_DATA': PATH_TO_DATA,
            'SHEET_NAME': SHEET_NAME,
            'X_CON_NAMES': X_CON_NAMES,
            'X_CAT_NAMES': X_CAT_NAMES,
            'Y_NAMES': Y_NAMES,
            'SKIP_ROWS': SKIP_ROWS,
            'path_to_results': path_to_results}

    # Create results directory ------------------------------------------------
    create_dir(path_to_results)

    # Copy this python script to results directory ----------------------------
    shutil.copy('ml_1_eda.py', path_to_results+'/ml_1_eda.py')

    # Load data ---------------------------------------------------------------
    # Load predictors from excel file
    x = pd.read_excel(PATH_TO_DATA,
                      sheet_name=SHEET_NAME,
                      header=0,
                      usecols=X_CON_NAMES+X_CAT_NAMES,
                      dtype='float',
                      skiprows=SKIP_ROWS)
    # Load targets from excel file
    y = pd.read_excel(PATH_TO_DATA,
                      sheet_name=SHEET_NAME,
                      header=0,
                      usecols=Y_NAMES,
                      dtype='float',
                      skiprows=SKIP_ROWS)

    # Drop rows with nans -----------------------------------------------------
    if task['DROP_NAN']:
        # Drop rows with nans
        x, y = drop_nan_rows(x, y)

    # Limit number of samples -------------------------------------------------
    # Subsample predictors
    x = x.sample(n=min(x.shape[0], task['MAX_SAMPLES']),
                 random_state=3141592,
                 ignore_index=False)
    # Slice targets to fit subsampled predictors
    y = y.loc[y.index, :].reset_index(drop=True)
    # Reset index of predictors
    x = x.reset_index(drop=True)

    # One-hot-encode categorical predictors -----------------------------------
    if task['X_CAT_NAMES']:
        # Instanciate one-hot-encoder
        ohe = OneHotEncoder(categories='auto',
                            drop='if_binary',
                            sparse_output=False,
                            dtype=int,
                            handle_unknown='error',
                            min_frequency=None,
                            max_categories=None).set_output(transform='pandas')
        # Fit-transform one-hot-encoder
        x_ohe = ohe.fit_transform(x[task['X_CAT_NAMES']])
        # Get one-hot-encoder category names
        task['x_ohe_names'] = list(ohe.get_feature_names_out())
        # All predictor names
        task['x_names'] = task['X_CON_NAMES']+task['x_ohe_names']
    else:
        # Pass on empty
        x_ohe = pd.DataFrame()
        task['x_ohe_names'] = []
        task['x_names'] = task['X_CON_NAMES']

    # Impute missing values in continous predictors ---------------------------
    if task['X_CON_NAMES']:
        # Instanciate imputer
        imp = KNNImputer(missing_values=np.nan,
                         n_neighbors=3,
                         weights='distance',
                         metric='nan_euclidean',
                         copy=True,
                         add_indicator=False).set_output(transform='pandas')
        # Impute missing values in continous data
        x_imp = imp.fit_transform(x[task['X_CON_NAMES']])
    else:
        # Pass on empty
        x_imp = pd.DataFrame()

    # Get predictors ----------------------------------------------------------
    # Predictors
    x = pd.concat([x_imp, x_ohe], axis=1)

    # Store data --------------------------------------------------------------
    # Save predictors
    x.to_excel(path_to_results+'/'+ANALYSIS_NAME+'_data_x.xlsx')
    # Save targets
    y.to_excel(path_to_results+'/'+ANALYSIS_NAME+'_data_y.xlsx')

    # Exploratory data analysis (EDA) -----------------------------------------
    # Run EDA
    eda(task, x, y)


if __name__ == '__main__':
    main()
