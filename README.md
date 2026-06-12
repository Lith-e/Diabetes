#Diabetes


DIABETES DETECTION USING 2 DATASET " MIMIC IV " and " PIMA INDIANS " 

Preprocessing:
1. LOF - To detect abnormal/noisy medical records.
2. Sparse encoder - data reconstruction

Data Handling Techniques:
    Method              Reason            
 Remove samples    cleanest data       
 KNN imputation    preserve similarity 
 Mean replacement  simple baseline     
 Zero replacement  comparison baseline 

Data Augumentation
SMOTE - Balancing the data through creating synthetic minority classes

Models 

AdaBoost -	handles difficult cases
Gradient - Boosting	strong nonlinear learning
HistGradientBoost - fast boosting
ExtraTrees - robust to noise
CatBoost	- handles categorical data
XGBoost	- powerful prediction
KNN	- local pattern learning

Ensemble Technique
Stacking Ensemble

 Feature Importance - to find which medical image matters most
 SHAP - to know which base model performs well
IT USES SMOT, LOF, SHAP 
