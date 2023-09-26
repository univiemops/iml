How to install with conda step-by-step (20230920)
(1)  Install anaconda (tested with 2023.03-1)
(2)  Open a anaconda prompt
(3)  Update conda: conda update conda
(4)  Setup a new environment: conda create -n mlenv
(5)  Activate the new environment: conda activate mlenv
(6)  Install python: conda install -c conda-forge python=3.11
(7)  Install spyder: conda install -c conda-forge spyder
(8)  Install ipywidgets: conda install -c conda-forge ipywidgets
(9)  Install matplotlib: conda install -c conda-forge matplotlib=3.7
(10) Install seaborn: conda install -c conda-forge seaborn
(11) Install openpyxl: conda install -c conda-forge openpyxl
(12) Install lightgbm: conda install -c conda-forge lightgbm
(13) Install shap: conda install -c conda-forge shap

How to install with less commands (20230920)
(1)  Install anaconda (tested with 2023.03-1)
(2)  Open a anaconda prompt
(3)  Update conda: conda update conda
(4)  Setup a new environment: conda create -n mlenv
(5)  Activate the new environment: conda activate mlenv
(6)  Install: conda install -c conda-forge python=3.11 spyder ipywidgets matplotlib=3.7 seaborn openpyxl lightgbm shap

How to update environment with conda (20230920)
(1)  Open a anaconda prompt
(2)  Update all packages in env: conda update -c conda-forge -n mlenv --all

How to get developer version of shap (20230920)
(1) Install: pip install git+https://github.com/shap/shap