# Decoupling-Deep-Domain-Adaptation-Method
Code files for "An Iterative Resampling Deep Decoupling Domain Adaptation Method for Class-imbalance Bearing Fault Diagnosis Under Variant Working Conditions"

1. CWRU-method
The content of this folder is the code files used to train the model using CWRU dataset. The structure of the dataset can refer to the description in the paper.
You can execute the file "main_iter_new.py" after modifying the data path. Hyperparameters can be modified and changed by argparse in CMD or shell script.

2. MBD-method
The content of this folder is the code files used to train the model using MBD dataset collected by our team. The execution way is consistent with the CWRU-method. MBD data is stored in folder "dataimba" and can be loaded correctly following the way of calling data in code in "main_iter_new.py".
