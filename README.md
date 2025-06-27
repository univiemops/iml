# Interpretable Machine Learning (iml)  
iml_1_eda.py, iml_2_mdl_lgbm.py, iml_2_mdl_tabpfn.py, iml_3_plt.py  
(20250627)  
  
Install via pip  
(1)  Install python from: https://www.python.org/  
(2)  Install Spyder IDE from: https://www.spyder-ide.org/  
(3)  Open cmd prompt  
(4)  Setup new environment: python -m venv iml\  
(5)  Activate new environment: iml\Scripts\activate  
(6)  Update pip: python -m pip install --upgrade pip  
(7)  Get packages: pip install spyder-kernels scikit-learn matplotlib seaborn openpyxl tabpfn shap lightgbm torch  
(8)  Optional CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade  
(9)  Add venv to spyder at Preferences->Python interpreter: C:\Users\*yourusername*\iml\Scripts\python.exe  
  
Workarounds for tree_explainer:  
(1) ExplainerError: The background dataset you provided does not cover all the leaves althought feature_perturbation="tree_path_dependent" and background dataset=None  
Add 'and self.data is not None' in line 346 of _tree.py  
(2) Case n_classes and interactions: Only one set of interaction is provided, but not a seperate per class  
Change 'if model.n_classes_ > 2:' to 'if model.n_classes_ >= 2:' in line 1295 of _tree.py  
  
