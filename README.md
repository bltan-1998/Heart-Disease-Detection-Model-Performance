# <span style="color:#e63939">Heart Disease Detection: Machine Learning Model Performance 
>**Aims**
>
>As precision medicine technology improves with development of AI, machine learning and deep learning algorithms have been extensively developed to help identifying and classifying patient outcomes under clinical settings. This project was started with the purpose to investigate model performance of different machine learning models from different packages in classifying heart disease from the database (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
>

**Author:** Tan Bee Ling


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

| Feature   | Type       | Description |
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

Using the computerized database from [1], database was inspected, cleaned by removing patients with incomplete information , convert categorical data into integers and change the feature for investigation from "num" to "class", sorted variables in order of categorical, continuous variables and binary heart disease class, and saved as an SQL based data registry. Data was inspected, sorted and cleaned using SQL version 5.7.44 (https://dev.mysql.com/downloads/installer/). 

**Patient Population**

Based on [4], the patient population consisted of 303 consecutive patients referred for coronary angiography at the Cleveland Clinic between May 1981 and September 1984. All patients underwent routine clinical evaluation, including medical history, physical examination, resting electrocardiography, serum cholesterol measurement, and fasting blood glucose measurement. In addition, following informed consent, patients underwent three noninvasive tests as part of the research protocol: exercise electrocardiography, thallium scintigraphy, and cardiac fluoroscopy. The study was designed to minimize work-up bias by recording historical information without knowledge of noninvasive or angiographic findings, while the noninvasive tests were analyzed without knowledge of the historical or angiographic results. Coronary angiograms were interpreted by a cardiologist who was blinded to the other test results. 

While inspecting and cleaning the computerized database, 6 patients with incomplete information (ie. presence of uninformed values on variables) were removed. Therefore, Table 1 includes only 297 patients with complete information for heart disease prediction in this work. The patients had a mean age of 54 years old by taking its floor, 201 (68.0%) were men and 96 (32%) were female. 137 of them were diagnosed with heart disease. None had a history or electrocardiographic evidence of previous myocardial infarction, known valvular disease, or cardiomyopathic disease.

<img width="609" height="945" alt="6165712586532393285" src="https://github.com/user-attachments/assets/a6f019b0-027d-4a2d-905f-60ca3b7fce2c" />
<img width="610" height="958" alt="6165712586532393286" src="https://github.com/user-attachments/assets/67364cbc-c88a-46d4-8e8a-bd94df79fed9" />

Table 1: Baseline characteristics of the patients diagnosed with and without heart disease used for heart disease prediction in this work

| Variables | Shapiro Wilk Normality Test Results   |
|:----------|:--------------------------------------|
| age       | 0.005424                              |
| trestbps  | 2.416e-06                             |
| chol      | 1.019e-08                             |
| thalach   | 9.044e-05                             |
| oldpeak   | < 2.2e-16                             |

Chart 1: Shapiro wilk normality test results

**Statistical & Data Analysis** 

Baseline characteristics (Table 1) are summarized with means and standard deviations for continuous variables, frequency and percentages for categorical variables. Group comparisons used the Wilcoxon rank-sum test for continuous variables, and either Pearson's Chi-squared test or Fisher's exact test for categorical variables, depending on which was appropriate.

Data analysis was performed by following these steps:

1. Data were inspected and sorted.
2. Data were cleaned by excluding patients with missing values for any variable.
3. Certain binary variables coded with multiple values were recoded as 1 and 0.
4. The heart disease classification was collapsed from a four-level scale (0 = healthy, 1–3 = increasing severity) into a
   binary outcome (0 = healthy, 1 = diagnosed with heart disease).
6. The cleaned dataset was compiled into an SQL-based data registry.
7. Correlation matrices and diagnostic density plots of the original, non-missing data were generated, and the Shapiro-Wilk
   test was applied to assess normality in the imputed continuous datasets.

**Selection of Heart Disease Predictors**

Predictors were selected from top 5 variables most highly correlated to the feature “class” (represented by "num"), which records healthy individuals as 0, diagnosed individuals as 1.

<img width="710" height="484" alt="corr" src="https://github.com/user-attachments/assets/eea6e15e-79f4-4f15-990d-e021913e8790" />

Figure 1: Correlation map of continuous variables is obtained from Spearman correlation test 

**Prediction of Heart Disease & Outcomes**

Based on previous experience and available literature, the following machine learning models were chosen to predict heart diasease outcome. All models were recomputed after hyper-parameter fine-tuning of all classification algorithms.
<img width="1192" height="451" alt="image" src="https://github.com/user-attachments/assets/74e7ea13-4ad8-4ac6-95ae-47d00567546e" />

The architecture and set up of the models are described as below 

**(1)Backward Propagation Neural Network (BPNN) Model Development**

The BPNN models were built and implemented using PyTorch version 2.7 (https://pytorch.org/blog/pytorch-2-7/). Three network architectures were investigated, consisting of 1, 3, and 6 hidden layers (n), respectively. Figure 2 illustrates the network architecture used in this work.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/99133259-d544-4617-8d1c-05558b5c581a" />

Figure 2: BPNN Architecture 

Model parameters were optimized using the Adam optimizer with a learning rate of 0.001. Binary Cross-Entropy Loss (BCELoss) was used as the objective function due to the involvement of a binary outcome in a prediction task. 

**(2) Logistic Regression**

A logistic regression model was developed using a generalized linear model (GLM) with a binomial distribution and logistic regression (logit) function with package stars in R.

**(3) Bayesian Logistic Regression**

A Bayesian logistic regression model was implemented using the Stan framework through the rstanarm package in R. 

* Prior Specification

Prior distributions were derived from the coefficients and standard errors obtained from the conventional logistic regression model.

For each coefficient:

$$
\beta_i \sim \mathrm{Normal}\left(\hat{\beta}_i\; SE(\hat{\beta}_i)\right)
$$

where $\hat{\beta}_i$ and $SE(\hat{\beta}_i)$ denote the estimated regression coefficient and its standard error, respectively.


**(4) K-Nearest Neighbours**

A K-Nearest Neighbours classifier was developed in R using the caret package. 

Model training employed 10-fold cross-validation with probability estimation enabled.
The optimal number of neighbours (k) was automatically selected through hyperparameter tuning with a search range of up to 10 candidate values.

**(5) Support Vector Machine**

A Support Vector Machine classifier with a Radial Basis Function (RBF) kernel was implemented in R using the caret package.

Model tuning was performed using 10-fold cross-validation, and candidate hyperparameter combinations were automatically explored through caret’s tuning procedure.

**(6) Extreme Gradient Boosting (XGB)**

An Extreme Gradient Boosting (XGBoost) classifier was implemented in R using the xgbTree algorithm within the caret framework.

Hyperparameter tuning was performed using 10-fold cross-validation with three candidate tuning configurations evaluated automatically by caret.

Data processing was done by normalizing continuous data and split the dataset into 2 seperated train-validate set with ratio of 4:1. Following data preprocessing and model specification,  models were trained using the designated training dataset, while an independent testing dataset was retained for final evaluation. 10-fold cross-validation was performed within the training dataset. The training data were divided into ten folds, with each fold iteratively used as the validation subset while the remaining folds were used for model training.

**Model Evaluation from Validation**

Input set of validation set is employed to the trained model, and then compared with the actual output set of the corresponding input. Prediction models’ performances were all assessed by area under the ROC (Receiver Operating Characteristic) curve (AUC-ROC), sensitivity, specificity, F1 Score, Youden’s J Statistics, prediction bias, precision, and accuracy on R version 4.3.1 (2023-06-16 ucrt)(https://cran.r-project.org/bin/windows/base/old/4.3.1/)with appropriate libraries (eg. rstanarm, brms, caret, dplyr, e1071, forcats, GGally, ggplot2, janitor, lubridate, pROC, purrr, readr, rstanarm, stringr, tibble, tidyr. tisyverse, xgboost, kernlab, kmed) and Python version 3.11.9 (https://www.python.org/downloads/release/python-3119/) on Visual Studio Code version 1.103.1,1.103.2, 1.104.1,1.104.2  with appropriate libraries (i.e. matplotlib, numpy, pandas, scikit-learn, seaborn, statistics, sklearn, Pytotch as "torch", torchvision).

# 📈Results: 
Among 297 patients, the mean age was 54.54+/-9.05 years, of whom 160 are control while 137 are diagnosed. 5 most key features (ca, thal, oldpeak, thalach, cp) from a total of 13 variables were chosen to train the models. XGB(AUC=0.888) outperformed BLR (AUC=0.882), BPNN6 (AUC=0.866), SVM (AUC=0.853) and KNN (AUC=0.853), BPNN3 (AUC=0.833), LR (AUC=0.84) and BPNN1 (AUC=0.817). The XGB model showed the highest accuracy (89.83%) with highest Youden’s J Statistics (0.7762), the SVM model was the most sensitive one (95.83%), LR and BLR showed the highest specificity (0.9714), and BLR showed the highest precision (95%). XGB model performed the best in overall due to its high AUC-ROC, accuracy, Youden’s J Statistics (0.7762), F1-Score (0.8696), reasonably high specificity (83.33%) and low prediction bias (0.0339) which showed best reliability in prediction, best detection of disease while performing reasonably well in predicting control. 

<img width="1106" height="277" alt="image" src="https://github.com/user-attachments/assets/8623e94f-62ba-445b-8f32-5ac392bf2a79" />

<img width="1098" height="462" alt="image" src="https://github.com/user-attachments/assets/78223592-9ae0-441b-b85a-6090f6c6dc03" />

# 🛠️Discussions

**Summary of findings**

Among the eight models, XGB performed best overall (AUC = 0.888, accuracy = 89.83%, Youden's J = 0.7762, F1 = 0.8696, prediction bias = 0.0339). However, AUC values across models ranged only from 0.817 to 0.888, and with a validation set of roughly 60 patients, significance differences between model performances such as XGB (0.888) versus BLR (0.882) are yet to be verified. 

**Interpretation (not ok)**
(XGB-BLR-BPNN6-SVM-KNN-LR-BPNN3-BPNN1)
The following explanations are hypotheses that this study was not designed to test. XGB may have led due to tree-based boosting which can capture non-linear effects and interactions among predictors such as ca, thal, oldpeak, thalach and cp, which linear models cannot. The BPNNs may have been limited by the small sample (n = 297), as neural networks generally need more data to train stably. 

The models also differed in their classification performance. SVM had the highest sensitivity (95.83%), whereas LR and BLR had the highest specificity (0.9714), and BLR the highest precision (95%). In a screening context, missing a diseased patient is usually costlier than a false alarm, so a high-sensitivity model such as SVM may be preferable despite XGB's better overall balance. Conversely, a high-specificity model may suit confirmatory settings where unnecessary invasive follow-up should be avoided. The best model therefore depends on the clinical use case.

LR, SVM and BPNN3 showed prediction bias above 0.1, suggesting systematic over- or under-prediction of one class. The cause is unknown; class imbalance, the small validation set, or the lack of probability calibration are possibilities to investigate.

**Challenges & Limitations**
**Limitations** 
•	Sample size: 

The dataset has 297 complete cases and the validation set is small, so it is unknown whether this sample size is sufficient or these rankings would hold in larger samples. Therefore, sample size determination test is needed to ensure sufficiency of sample size for this dataset for reliable outcome classification [6]. One way is to perform a posteriori sample size calculation with a learning curve approach [5,6] to determine if this sample size is sufficiently large for model derivation.

•	Predictor selection: 

Predictors were chosen by correlation with the outcome. Correlation reflects association between variables, which is not powerful enough to show importance or causal influence between variables, and it is unknown whether the chosen five are optimal. As a results, feature selection approaches for machine learning based disease risk prediction should be employed before proceeding model classification to carefully determine and evaluate their importance in the prediction[7]. 

**Challanges** 
•	Limited predictors and data source:

Only 13 variables were available, from a single centre (Cleveland Clinic, 1981-1984). Generalisability to contemporary or other populations is unknown.  

**Future Work and Clinical Implications**
We plan to address the limitations as mentioned in discussion for model classification in future work.  Further steps include sample size calibration analysis, confidence intervals for all metrics, and exploration with medical institutions for larger datasets with more biomarkers and omics data if needed and accessible. These models are intended to support, not replace, clinical decision-making.

# 📮Conclusion: 
This study highlights the potential of machine learning models to support clinicians in CVD diagnosis using routinely available patient data.  All eight models achieved strong discriminative performance (AUC > 0.80), confirming their capability in identifying positive and negative class, however some models (LR, SVM, BPNN3) tend to make slightly unrealistic predictions due to their prediction bias > 0.1. Among them, XGB consistently outperformed, offering robust predictive reliability without much bias toward either class. Conversely, BPNN3 showed reduced sensitivity and a marked specificity-sensitivity value gap, limiting its diagnostic reliability relative to other models. These findings suggest that ensemble-based approaches such as XGB may provide the most effective framework for developing decision-support systems in CVD diagnosis. However, limited data size may lead to overfitting, thus, model performance needs to be further evaluated with larger datasets.

# 📋Database
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# 📒References:
1) Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
2) Souza, Cezar & Barreto, Cephas & Macedo, Lhayana & Oliveira de Brito, Bruna Alice & Targino, Victor & Betcel, Emanuel & Gomes de Almeida, Fernando & Rodrigues, Arthur & Malaquias, Ramon & Barroca Filho, Itamir. (2023). A systematic literature review on Machine Learning Model evaluation on healthcare applications. Research Society and Development. 12. e5412642042. 10.33448/rsd-v12i6.42042.
3) Aggarwal, Charu. (2018). Neural Networks and Deep Learning: A Textbook. 10.1007/978-3-319-94463-0.
4) Detrano R, Janosi A, Steinbrunn W, Pfisterer M, Schmid JJ, Sandhu S, Guppy KH, Lee S, Froelicher V. International application of a new probability algorithm for the diagnosis of coronary artery disease. Am J Cardiol. 1989 Aug 1;64(5):304-10. doi: 10.1016/0002-9149(89)90524-9. PMID: 2756873.
5)Böhm A, Segev A, Jajcay N, Krychtiuk KA, Tavazzi G, Spartalis M, Kollarova M, Berta I, Jankova J, Guerra F, Pogran E, Remak A, Jarakovic M, Sebenova Jerigova V, Petrikova K, Matetzky S, Skurk C, Huber K, Bezak B. Machine learning-based scoring system to predict cardiogenic shock in acute coronary syndrome. Eur Heart J Digit Health. 2025 Jan 6;6(2):240-251. doi: 10.1093/ehjdh/ztaf002. PMID: 40110217; PMCID: PMC11914733.
6) van Smeden M, Heinze G, Van Calster B, Asselbergs FW, Vardas PE, Bruining N, de Jaegere P, Moore JH, Denaxas S, Boulesteix AL, Moons KGM. Critical appraisal of artificial intelligence-based prediction models for cardiovascular disease. Eur Heart J. 2022 Aug 14;43(31):2921-2930. doi: 10.1093/eurheartj/ehac238. PMID: 35639667; PMCID: PMC9443991.
7) Pudjihartono N, Fadason T, Kempa-Liehr AW, O'Sullivan JM. A Review of Feature Selection Methods for Machine Learning-Based Disease Risk Prediction. Front Bioinform. 2022 Jun 27;2:927312. doi: 10.3389/fbinf.2022.927312. PMID: 36304293; PMCID: PMC9580915.
8) Yang, K., Liu, L. & Wen, Y. The impact of Bayesian optimization on feature selection. Sci Rep 14, 3948 (2024). https://doi.org/10.1038/s41598-024-54515-w.


