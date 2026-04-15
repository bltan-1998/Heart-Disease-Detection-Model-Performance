# Heart Disease Detection: Machine Learning Model Performance 
As precision medicine technology improves with development of AI, machine learning and deep learning algorithms have been extensively developed to help identifying and classifying patient outcomes under clinical settings. This project was started with the purpose to investigate model performance of different machine learning models from different packages in classifying heart disease from the database (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). 

# Author:
Tan Bee Ling

# Methods: 
1) Creating SQL data registry from the Heart Disease database [1] after data cleaning.
   
2) Features selection:
- Selecting top 5 variables most highly correlated to the feature “class”, which records healthy individuals as 0, diagnosed
  individuals as 1.
- Two correlation matrices using Pearson and Spearman correlation analysis.

3) Data Normalization:
- Data were all normalised to z-score.
- Train and test data were divided with the proportion of 4:1.

4) Training
- No resampling method was performed due to well distributed diagnosed and control ratio.
- Find the optimal threshold for prediction model prediction from the highest Youden’s J score:
  
  a) over 10-fold cross-validation in the train data on the SVM, BLR, LR, XGB and KNN models from packages for R,
  b) over 1000 epochs for the 3 NN models developed with PyTorch.

[Machine Learning Models implemented on R and Python]
<img width="1192" height="451" alt="image" src="https://github.com/user-attachments/assets/74e7ea13-4ad8-4ac6-95ae-47d00567546e" />

5) Model Evaluation from test results 
Prediction models’ performances were all assessed by area under the ROC (Receiver Operating Characteristic) curve (AUC-ROC), sensitivity, specificity, F1 Score, Youden’s J Statistics, prediction bias, precision, and accuracy.

# Results: 
Among 297 patients, the mean age was 54.54+/-9.05 years, of whom 160 are control while 137 are diagnosed. 5 most key features (ca, thal, oldpeak, thalach, cp) from a total of 13 variables were chosen to train the models. XGB(AUC=0.888) outperformed BLR (AUC=0.882), BPNN6 (AUC=0.866), SVM (AUC=0.853) and KNN (AUC=0.853), BPNN3 (AUC=0.833), LR (AUC=0.84) and BPNN1 (AUC=0.817). The XGB model showed the highest accuracy (89.83%) with highest Youden’s J Statistics (0.7762), the SVM model was the most sensitive one (95.83%), LR and BLR showed the highest specificity (0.9714), and BLR showed the highest precision (95%). XGB model performed the best in overall due to its high AUC-ROC, accuracy, Youden’s J Statistics (0.7762), F1-Score (0.8696), reasonably high specificity (83.33%) and low prediction bias (0.0339) which showed best reliability in prediction, best detection of disease while performing reasonably well in predicting control. 

<img width="1106" height="277" alt="image" src="https://github.com/user-attachments/assets/8623e94f-62ba-445b-8f32-5ac392bf2a79" />

<img width="1098" height="462" alt="image" src="https://github.com/user-attachments/assets/78223592-9ae0-441b-b85a-6090f6c6dc03" />

# Conclusion: 
This study highlights the potential of machine learning models to support clinicians in CVD diagnosis using routinely available patient data.  All eight models achieved strong discriminative performance (AUC > 0.80), confirming their capability in identifying positive and negative class, however some models (LR, SVM, BPNN3) tend to make slightly unrealistic predictions due to their prediction bias > 0.1. Among them, XGB consistently outperformed, offering robust predictive reliability without much bias toward either class. Conversely, BPNN3 showed reduced sensitivity and a marked specificity-sensitivity value gap, limiting its diagnostic reliability relative to other models. These findings suggest that ensemble-based approaches such as XGB may provide the most effective framework for developing decision-support systems in CVD diagnosis. However, limited data size may lead to overfitting, thus, model performance needs to be further evaluated with larger datasets.

# Database
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# References:
1) Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
2) Souza, Cezar & Barreto, Cephas & Macedo, Lhayana & Oliveira de Brito, Bruna Alice & Targino, Victor & Betcel, Emanuel & Gomes de Almeida, Fernando & Rodrigues, Arthur & Malaquias, Ramon & Barroca Filho, Itamir. (2023). A systematic literature review on Machine Learning Model evaluation on healthcare applications. Research Society and Development. 12. e5412642042. 10.33448/rsd-v12i6.42042.
3) Aggarwal, Charu. (2018). Neural Networks and Deep Learning: A Textbook. 10.1007/978-3-319-94463-0.


