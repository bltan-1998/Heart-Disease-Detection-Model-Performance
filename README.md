# <span style="color:#e63939">Heart Disease Detection: Machine Learning Model Performance 
> As precision medicine technology improves with development of AI, machine learning and deep learning algorithms have been extensively developed to help identifying and classifying patient outcomes under clinical settings. This project was started with the purpose to investigate model performance of different machine learning models from different packages in classifying heart disease from the database (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). 

**Author:** Tan Bee Ling

# 📌Abstract
(Add in)

**Key Findings:**

# 📊Dataset
| Property           | Details                               |
|:-------------------|:--------------------------------------|
| Source             | UCI Machine Learning Repository       |
| Rows               | 303                                   |
| Features           | 13                                    |
| Target             | Binary                                |
| Class Balance      | xx                                    |
| Missing values     | NA                                    |

📲**Feature Descriptions**

| Feature   | Type        | Description |
|-----------|------------|-------------|
| Age       | Numerical  | Patient’s age (in years) |
| Sex       | Binary     | Gender (1 = Male, 0 = Female) |
| cp        | Categorical| Chest pain type (0–3): 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
| trestbps  | Numerical  | Resting blood pressure (mm Hg) |
| chol      | Numerical  | Serum cholesterol (mg/dl) |
| fbs       | Binary     | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| restecg   | Categorical| Resting ECG results (0–2): 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy (Estes’ criteria) |
| thalach   | Numerical  | Maximum heart rate achieved |
| exang     | Binary     | Exercise-induced angina (1 = Yes, 0 = No) |
| oldpeak   | Numerical  | ST depression induced by exercise relative to rest |
| slope     | Categorical| Slope of peak exercise ST segment (0–2): 0 = Upsloping, 1 = Flat, 2 = Downsloping |
| ca        | Numerical  | Number of major vessels (0–3) colored by fluoroscopy |
| thal      | Categorical| Thalassemia status: 0 = Unknown/Null, 1 = Normal, 2 = Fixed defect, 3 = Reversible defect |
| num       | Binary     | Target variable (1 = Heart disease, 0 = Healthy) |

**🔍Exploratory Data Analysis**
(Relationship)
<img width="849" height="822" alt="image" src="https://github.com/user-attachments/assets/c041ea21-a1b6-4867-a986-f1a14569f82b" />

# ⚙️Methodology: 

Machine Learning Models implemented on R and Python
<img width="1192" height="451" alt="image" src="https://github.com/user-attachments/assets/74e7ea13-4ad8-4ac6-95ae-47d00567546e" />

1) Creating SQL data registry from the Heart Disease database [1] after data cleaning.
   
2) Features selection:
- Selecting top 5 variables most highly correlated to the feature “class” (represented by "num"), which records healthy
  individuals as 0, diagnosed individuals as 1.
- Two correlation matrices using Pearson and Spearman correlation analysis.

3) Data Normalization:
- Data were all normalised to z-score.
- Train and test data were divided with the proportion of 4:1.

4) Model Training
- Find the optimal threshold for prediction model prediction from the highest Youden’s J score:
  
  a) over 10-fold cross-validation in the train data on the SVM, BLR, LR, XGB and KNN models from packages for R,
  
  b) over 1000 epochs for the 3 NN models developed with PyTorch.

5) Model Evaluation from test results 
Prediction models’ performances were all assessed by area under the ROC (Receiver Operating Characteristic) curve (AUC-ROC), sensitivity, specificity, F1 Score, Youden’s J Statistics, prediction bias, precision, and accuracy.

# 📈Results: 
Among 297 patients, the mean age was 54.54+/-9.05 years, of whom 160 are control while 137 are diagnosed. 5 most key features (ca, thal, oldpeak, thalach, cp) from a total of 13 variables were chosen to train the models. XGB(AUC=0.888) outperformed BLR (AUC=0.882), BPNN6 (AUC=0.866), SVM (AUC=0.853) and KNN (AUC=0.853), BPNN3 (AUC=0.833), LR (AUC=0.84) and BPNN1 (AUC=0.817). The XGB model showed the highest accuracy (89.83%) with highest Youden’s J Statistics (0.7762), the SVM model was the most sensitive one (95.83%), LR and BLR showed the highest specificity (0.9714), and BLR showed the highest precision (95%). XGB model performed the best in overall due to its high AUC-ROC, accuracy, Youden’s J Statistics (0.7762), F1-Score (0.8696), reasonably high specificity (83.33%) and low prediction bias (0.0339) which showed best reliability in prediction, best detection of disease while performing reasonably well in predicting control. 

<img width="1106" height="277" alt="image" src="https://github.com/user-attachments/assets/8623e94f-62ba-445b-8f32-5ac392bf2a79" />

<img width="1098" height="462" alt="image" src="https://github.com/user-attachments/assets/78223592-9ae0-441b-b85a-6090f6c6dc03" />

# 🛠️Discussions
**Limitations**
- Limited size of sample in this database to further train, validate and test the machine learning models
- Limited number of variables available as predictors
- Limited information on relationship between predictors and their link to the presence of heart disease

  
**Future Work**

To address the above limitations, Bayesian network with causal relationship information between the predictors themselves and the predictors and heart disease will be crafted from the utility of Bayesian statistics and probability theory. Besides, further collaboration with medical institutions is necessary for larger size of database with more biomarkers and omics available.

# 📮Conclusion: 
This study highlights the potential of machine learning models to support clinicians in CVD diagnosis using routinely available patient data.  All eight models achieved strong discriminative performance (AUC > 0.80), confirming their capability in identifying positive and negative class, however some models (LR, SVM, BPNN3) tend to make slightly unrealistic predictions due to their prediction bias > 0.1. Among them, XGB consistently outperformed, offering robust predictive reliability without much bias toward either class. Conversely, BPNN3 showed reduced sensitivity and a marked specificity-sensitivity value gap, limiting its diagnostic reliability relative to other models. These findings suggest that ensemble-based approaches such as XGB may provide the most effective framework for developing decision-support systems in CVD diagnosis. However, limited data size may lead to overfitting, thus, model performance needs to be further evaluated with larger datasets.

# 📋Database
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# 📒References:
1) Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
2) Souza, Cezar & Barreto, Cephas & Macedo, Lhayana & Oliveira de Brito, Bruna Alice & Targino, Victor & Betcel, Emanuel & Gomes de Almeida, Fernando & Rodrigues, Arthur & Malaquias, Ramon & Barroca Filho, Itamir. (2023). A systematic literature review on Machine Learning Model evaluation on healthcare applications. Research Society and Development. 12. e5412642042. 10.33448/rsd-v12i6.42042.
3) Aggarwal, Charu. (2018). Neural Networks and Deep Learning: A Textbook. 10.1007/978-3-319-94463-0.


