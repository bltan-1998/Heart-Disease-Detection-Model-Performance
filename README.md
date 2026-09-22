# <span style="color:#e63939">Heart Disease Detection: Machine Learning Model Performance 
>**Aims**
>
>As precision medicine technology improves with development of AI, machine learning and deep learning algorithms have been extensively developed to help identifying and classifying patient outcomes under clinical settings. This project was started with the purpose to investigate model performance of different machine learning models from different packages in classifying heart disease from the database (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
>
>**Methods & Results**
>
>**Conclusion**

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


# ⚙️Methodology: 

**Data Collection & Data Registry Creation** 

Based on [4], the data collection protocol included routine clinical assessment and three noninvasive cardiovascular investigations. Clinical information comprised medical history, physical examination, resting electrocardiography, serum cholesterol, and fasting blood glucose measurements. Following informed consent, patients underwent exercise electrocardiography, exercise thallium scintigraphy, and cardiac fluoroscopy to assess coronary calcium.

13 clinical and test variables were considered, which comprised four clinical variables: age, sex, chest pain type, and systolic blood pressure. Additional routine clinical and laboratory variables included serum cholesterol, fasting blood glucose >120 mg/dL, and resting ECG findings. Exercise and noninvasive test variables included maximum heart rate, exercise-induced angina, ST-segment slope, ST-segment depression, exercise thallium scintigraphy findings, and the number of major coronary vessels showing calcium on fluoroscopy.

To minimize potential work-up bias, information from different stages of the assessment was collected and interpreted independently [4]. Historical information was recorded without knowledge of the noninvasive test or angiographic results, while the noninvasive tests were analyzed without knowledge of the patient's history or angiographic findings [4]. Coronary angiograms were interpreted by a cardiologist who was blinded to the other test results [4].

The collected variables were subsequently entered into a computerized database [4]. Coronary artery disease status was determined from coronary angiography, with an angiogram classified as abnormal when there was greater than 50% diameter narrowing of a major coronary vessel [4]. This angiographic disease status, represented as "num", was used as the dependent outcome variable for development of the prediction model in this work [4].

Using the computerized database from [1], database was cleaned, sorted and saved as an SQL based data registry, which is represented in chart 1.

(chart 1: Data cleaning: Remove unwanted data which means with missing info, left with 297 patients, then reorganize variables : categorical> cts var > num), save as database in SQL version (...)


**Patient Population**

Based on [4], the patient population consisted of 303 consecutive patients referred for coronary angiography at the Cleveland Clinic between May 1981 and September 1984. The patients had a mean age of 54 years, and 206 (68.0%) were men. None had a history or electrocardiographic evidence of previous myocardial infarction, known valvular disease, or cardiomyopathic disease.

All 303 patients underwent routine clinical evaluation, including medical history, physical examination, resting electrocardiography, serum cholesterol measurement, and fasting blood glucose measurement. In addition, following informed consent, patients underwent three noninvasive tests as part of the research protocol: exercise electrocardiography, thallium scintigraphy, and cardiac fluoroscopy. The study was designed to minimize work-up bias by recording historical information without knowledge of noninvasive or angiographic findings, while the noninvasive tests were analyzed without knowledge of the historical or angiographic results. Coronary angiograms were interpreted by a cardiologist who was blinded to the other test results. 

However, 6 patients with incomplete information (ie. presence of uninformed values on variables) were removed during data cleaning. Therefore, Table 1 includes only 297 patients with complete information for heart disease prediction in this work.

<img width="609" height="945" alt="6165712586532393285" src="https://github.com/user-attachments/assets/a6f019b0-027d-4a2d-905f-60ca3b7fce2c" />
<img width="610" height="958" alt="6165712586532393286" src="https://github.com/user-attachments/assets/67364cbc-c88a-46d4-8e8a-bd94df79fed9" />

Table 1: Baseline characteristics of the patients diagnosed with and without heart disease used for heart disease prediction in this work

**Statistical & Data Analysis**

As shown in baseline characteristics (Table 1), continuous variables are presented as means and standard deviations, whereas categorical variables are presented as percentages.

**Selection of Heart Disease Predictors**

Predictors were selected from top 5 variables most highly correlated to the feature “class” (represented by "num"), which records healthy individuals as 0, diagnosed individuals as 1.

<img width="710" height="484" alt="corr" src="https://github.com/user-attachments/assets/eea6e15e-79f4-4f15-990d-e021913e8790" />

Figure 1: Correlation map of continuous variables is obtained from Spearman correlation test 

**Prediction of Heart Disease & Outcomes**

Based on previous experience and available literature, the following machine learning models were chosen to predict heart diasease outcome. 
<img width="1192" height="451" alt="image" src="https://github.com/user-attachments/assets/74e7ea13-4ad8-4ac6-95ae-47d00567546e" />

All models were recomputed after hyper-parameter fine-tuning of all classification algorithms.

The models were set up and implemented on R and Python as follows

**(1) Model Setup**

3) Data Normalization:
- Data were all normalised to z-score.
- Train and test data were divided with the proportion of 4:1.

4) Model Training
- Find the optimal threshold for prediction model prediction from the highest Youden’s J score:
  
  a) over 10-fold cross-validation in the train data on the SVM, BLR, LR, XGB and KNN models from packages for R,
  
  b) over 1000 epochs for the 3 NN models developed with PyTorch.

5) Model Evaluation from test results 
Prediction models’ performances were all assessed by area under the ROC (Receiver Operating Characteristic) curve (AUC-ROC), sensitivity, specificity, F1 Score, Youden’s J Statistics, prediction bias, precision, and accuracy.

**Validation**


# 📈Results: 
Among 297 patients, the mean age was 54.54+/-9.05 years, of whom 160 are control while 137 are diagnosed. 5 most key features (ca, thal, oldpeak, thalach, cp) from a total of 13 variables were chosen to train the models. XGB(AUC=0.888) outperformed BLR (AUC=0.882), BPNN6 (AUC=0.866), SVM (AUC=0.853) and KNN (AUC=0.853), BPNN3 (AUC=0.833), LR (AUC=0.84) and BPNN1 (AUC=0.817). The XGB model showed the highest accuracy (89.83%) with highest Youden’s J Statistics (0.7762), the SVM model was the most sensitive one (95.83%), LR and BLR showed the highest specificity (0.9714), and BLR showed the highest precision (95%). XGB model performed the best in overall due to its high AUC-ROC, accuracy, Youden’s J Statistics (0.7762), F1-Score (0.8696), reasonably high specificity (83.33%) and low prediction bias (0.0339) which showed best reliability in prediction, best detection of disease while performing reasonably well in predicting control. 

<img width="1106" height="277" alt="image" src="https://github.com/user-attachments/assets/8623e94f-62ba-445b-8f32-5ac392bf2a79" />

<img width="1098" height="462" alt="image" src="https://github.com/user-attachments/assets/78223592-9ae0-441b-b85a-6090f6c6dc03" />

# 🛠️Discussions
**Limitations** (The following should be amended to be stated as not known if it is enough instead of saying insufficient because we do not have enough proof to show its sufficiency. Therefore we can say that we will move on to check if they are sufficient. Besides, limitations should also include what we find problematic. There might have something which happens against our expectations, if we do not know any, we can also state something we have not really discovered. But one thing for sure is that we can check the model performance, and tell what might be the problems, or if some figures seem bad, then we can propose what are they, of course we should mention that our predictors choice is not so appropriate with correlation matrix as it imples correlation without telling us importance, where we need further classification using weightage approaches which then can tell us which are more important, inclusing bayesian statistics to provide some information of causal relationship.)
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
4) Detrano R, Janosi A, Steinbrunn W, Pfisterer M, Schmid JJ, Sandhu S, Guppy KH, Lee S, Froelicher V. International application of a new probability algorithm for the diagnosis of coronary artery disease. Am J Cardiol. 1989 Aug 1;64(5):304-10. doi: 10.1016/0002-9149(89)90524-9. PMID: 2756873.


