iml_1_eda.py, iml_2_mdl.py, iml_3_plt.py
(20240320)

Install
(1)  Install git from: https://www.git-scm.com/downloads
(2)  Install miniconda from: https://docs.conda.io/projects/miniconda/en/latest/
(3)  Open anaconda prompt
(4)  Update conda: conda update conda
(5)  Setup new environment: conda create -n iml
(6)  Activate new environment: conda activate iml
(7)  Get packages: conda install -c conda-forge -n iml python=3.11 scikit-learn spyder ipywidgets matplotlib seaborn openpyxl lightgbm shap=*=cpu* blas=2.120=mkl

Update
(1)  Open anaconda prompt
(2)  Update packages: conda update -c conda-forge -n iml --all shap=*=cpu* blas=2.120=mkl

Developer version of shap
(1)  Get shap: pip install git+https://github.com/shap/shap

Workaround for error in SHAP computation in case of multiclass and interactions=True
(1) Add 'and self.data is not None' in line 346 of _tree.py