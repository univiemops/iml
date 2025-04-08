# Interpretable Machine Learning (iml)  
iml_1_eda.py, iml_2_mdl_lgbm.py, iml_2_mdl_tabpfn.py, iml_3_plt.py  
(20250408)  
  
Install via pip  
(1)  Install python from: https://www.python.org/  
(2)  Open cmd prompt  
(3)  Setup new environment: py -m venv iml\  
(4)  Activate new environment: iml\Scripts\activate  
(5)  Update pip: py -m pip install --upgrade pip  
(6)  Get packages: pip install spyder ipywidgets scikit-learn matplotlib seaborn openpyxl tabpfn shap lightgbm  
(7)  Optional CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade  
  
Workarounds for tree_explainer:  
(1) ExplainerError: The background dataset you provided does not cover all the leaves althought feature_perturbation="tree_path_dependent" and background dataset=None  
Add 'and self.data is not None' in line 346 of _tree.py  
(2) Case n_classes and interactions: Only one set of interaction is provided, but not a seperate per class  
Change 'if model.n_classes_ > 2:' to 'if model.n_classes_ >= 2:' in line 1295 of _tree.py  
  
