How to install with conda step-by-step (20230320)
(1)  Install anaconda or miniconda
(Bugfix) Go to anaconda3>Library>bin, search and copy following dll files libcrypto-1_1-x64.dll libssl-1_1-x64.dll and paste to anaconda3>DLLs
(2)  Open a anaconda prompt with admin rights
(3)  Setup a new environment: conda create -n mlenv
(4)  Activate the new environment: conda activate mlenv
(5)  Install Python: conda install -c conda-forge python=3.10
(6)  Install Spyder IDE: conda install -c conda-forge spyder
(7)  Install ipywidgets: conda install -c conda-forge ipywidgets
(8)  Install matplotlib version 3.5: conda install -c conda-forge matplotlib=3.5
(9)  Install seaborn: conda install -c conda-forge seaborn
(10) Install openpyxl: conda install -c conda-forge openpyxl
(11) Install more-itertools: conda install -c conda-forge more-itertools
(12) Install SHAP: conda install -c conda-forge shap
(13) Install lightgbm: conda install -c conda-forge lightgbm

How to install with conda one command (20230320)
(1)  Install anaconda or miniconda
(Bugfix) Go to anaconda3>Library>bin, search and copy following dll files libcrypto-1_1-x64.dll libssl-1_1-x64.dll and paste to anaconda3>DLLs
(2)  Open a anaconda prompt with admin rights
(3)  Setup a new environment: conda create -n mlenv
(4)  Activate the new environment: conda activate mlenv
(5)  Install: conda install -c conda-forge python=3.10 spyder ipywidgets matplotlib=3.5 seaborn openpyxl more-itertools shap lightgbm

How to update environment with conda (20230320)
(1) Open a anaconda prompt with admin rights
(2) Update all packages in env: conda update -c conda-forge -n mlenv --all

Optional: Speed up for AMD Ryzen processors (20230320)
(1) Open a anaconda prompt with admin rights
(2) Activate the environment: conda activate mlenv
(3) Install old mkl version: conda install -c conda-forge mkl=2020.0
(4) Add environment variable: MKL_DEBUG_CPU_TYPE=5