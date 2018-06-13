# Introduction
Even with an annual expenditure of more than $3 trillion, the U.S. healthcare system is far from optimal. For example, the third leading cause of death in the U.S. is preventable medical error. Computer based Decision Support Systems have been proposed to take care of the error to a reasonable extent. However, such systems have not been implemented to utilize the patient data obtained on a daily basis. Another big challenge is in the non-uniformity of diagnosis offered by doctors. The standard deviation in the doctor prescriptions still remains large.

A cardiac arrhythmia is any abnormal heart rate or rhythm. It can be classified into different classes. Some cases of arrhythmia can be critical and only with quick response can the patient reduce risks of complications. The diagnosis of arrhythmia involves handling of huge amount of ECG data which poses the risk of human error in the interpretation of data. If a health care provider failed to diagnose a severe case of arrhythmia, they can even be held liable for medical malpractice. Hence, computer assisted analysis of the ECG data and arrhythmia detection and classification can play a huge role as a decision support system to the doctors. Following are some of the facts which show the importance of the problem:

1. One study showed that up to 23% of misdiagnosed heart attacks were due to the improper reading of a patient's ECG
2. One study showed that as compared to electrocardiographers, the physicians misread as normal about 36% of abnormal T waves
3. One study showed that out of 1.5 million heart attacks occurring every year in United States, 11000 cases are because of misdiagnoses
4. About 4.7 billion dollars paid in 2008 to resolve *all* malpractice claims nationwide

# Objective:

Machine learning algorithms are extensively used in developing decision support systems in medical field.
Through this project, we intend to use supervised machine learning algorithms to analyse the various features of the electrocardiogram (ECG) of the patients and other physiological characteristics. The Cardiac Arrhythmia data set publicly available on UCI Machine Learning Repository has been utilized for developing the classifiers. The objective of this project to build 2 different models.
a. Model 1- Model for detecting the presence of Arrythmia.
b. Model 2- Model for classifying Arrythmia into different classes.

# Scope of Work:
The scope of this project comprises of the following aspects:

### 1. Data cleaning:
To replace the garbage values with null values. Further, impute the null values using column mean.

### 2. Feature selection by Boruta package:
To find significant predictors to be used for building the models.

### 3. Classification by various statistical learning methods:
To employ various supervised learning algorithms for achieving the aforementioned objectives.

### 4. Comparison of results:
To compare the results obtained from various models and understand the behaviour of different algorithms, and further use it as an application.

# Approach:

## Data Description:
The data belongs to the ECG readings and some of the physical description of 451 patients.
There are a total of 279 attributes (206 linear and 73 nominal) and a single output which is categorical. Each of 451 patients are divided into 16 classes based on the value of their attributes.

