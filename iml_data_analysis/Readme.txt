Interpretable Machine Learning (iml)
iml_1_eda.py, iml_2_mdl_lgbm.py, iml_2_mdl_tabpfn.py, iml_3_plt.py
(20250716)

Install via pip
(1)  Install miniconda from: https://www.anaconda.com/
(2)  Optional if nvidia: install cuda from: https://developer.nvidia.com/cuda-downloads
(3)  Optional if nvidia: install cudnn from: https://developer.nvidia.com/cudnn
(4)  Open Anaconda Prompt
(5)  Setup new environment: conda create --name iml
(6)  Activate new environment: conda activate iml
(7)  Install python and pip via conda: conda install python=3.13 pip
(8)  Get tabpfn via pip: pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
(9)  Get other packages via pip: pip install spyder ipywidgets scikit-learn matplotlib seaborn openpyxl shap lightgbm
(10) Optional if nvidia via pip: pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade

Workarounds for tree_explainer:
(1) ExplainerError: The background dataset you provided does not cover all the leaves althought feature_perturbation="tree_path_dependent" and background dataset=None
Open miniconda3\envs\iml\Lib\site-packages\shap\explainers\_tree.py
Add 'and self.data is not None:' in line 467 after 'if self.feature_perturbation == "tree_path_dependent"'
(2) Case n_classes and interactions: Only one set of interaction is provided, but not a seperate per class
Open miniconda3\envs\iml\Lib\site-packages\shap\explainers\_tree.py
Change 'if model.n_classes_ > 2:' to 'if model.n_classes_ >= 2:' in line 1336