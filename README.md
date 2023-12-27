# IML data analysis
Interpretable machine-learning data analysis  
  
How to install (20231227)  
(1)  Install miniconda (tested with 2023.09-0)  
(2)  Open a anaconda prompt  
(3)  Update conda: conda update conda  
(4)  Setup a new environment: conda create -n iml  
(5)  Activate the new environment: conda activate iml  
(6)  Install: conda install -c conda-forge -n iml python=3.11 spyder ipywidgets matplotlib seaborn openpyxl lightgbm shap blas=*=*mkl  
  
How to update environment with conda (20231227)  
(1)  Open a anaconda prompt  
(2)  Update all packages in env: conda update -c conda-forge -n iml --all blas=*=*mkl  
