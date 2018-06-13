# Introduction
Even with an annual expenditure of more than $3 trillion, the U.S. healthcare system is far from optimal. For example, the third leading cause of death in the U.S. is preventable medical error. Computer based Decision Support Systems have been proposed to take care of the error to a reasonable extent. However, such systems have not been implemented to utilize the patient data obtained on a daily basis. Another big challenge is in the non-uniformity of diagnosis offered by doctors. The standard deviation in the doctor prescriptions still remains large.

A cardiac arrhythmia is any abnormal heart rate or rhythm. It can be classified into different classes. Some cases of arrhythmia can be critical and only with quick response can the patient reduce risks of complications. The diagnosis of arrhythmia involves handling of huge amount of ECG data which poses the risk of human error in the interpretation of data. If a health care provider failed to diagnose a severe case of arrhythmia, they can even be held liable for medical malpractice. Hence, computer assisted analysis of the ECG data and arrhythmia detection and classification can play a huge role as a decision support system to the doctors. Following are some of the facts which show the importance of the problem:

1. One study showed that up to 23% of misdiagnosed heart attacks were due to the improper reading of a patient's ECG
2. One study showed that as compared to electrocardiographers, the physicians misread as normal about 36% of abnormal T waves
3. One study showed that out of 1.5 million heart attacks occurring every year in United States, 11000 cases are because of misdiagnoses
4. About 4.7 billion dollars paid in 2008 to resolve *all* malpractice claims nationwide

# Objective:

Machine learning algorithms are extensively used in developing decision support systems in medical field.
Through this project, we intend to use supervised machine learning algorithms to analyse the various features of the electrocardiogram (ECG) of the patients and other physiological characteristics. The Cardiac Arrhythmia data set publicly available on UCI Machine Learning Repository has been utilized for developing the classifiers. 
The objective of this project to build 2 different models.

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

![Image1](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image1.PNG)

![Image2](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image2.PNG)

## Reasoning of Approach:

We have implemented models for two purposes.

  a. Detection of cardiac arrhythmia
  b. Classification of cardiac arrhythmia.
  
In the model for detection, the data points have been classified into two classes- “Normal” & “Arrhythmia”. This model only identifies if the patient is normal (class 1) or suffers from any form of arrhythmia (class 2 to 16).
The model for classification classified the patient into one of 16 classes, with class 1 representing normal and classes 2 to 16 representing a condition of cardiac arrhythmia. The arrhythmia class will be treated as the ‘Positive’ class.
The following describes the step-by-step approach for implementing the two models.

### Data Cleaning

The data set comprises of 279 attributes of 451 patients. The attributes are as indicated in the section above.
As the first step, the missing values were replaced with null values. Then the columns with same value in all the 451 rows were identified. 17 attributes were found to have a single value for all the data points. As these columns do not explain any variation between the data points and response, they are irrelevant and need not be considered for further analysis. Hence, they were deleted from the data set.

In the next step, we identified the missing values in the data set.
It was observed that five columns had missing values.

![Image3](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image3.PNG)

The columns J_Angle and P_Angle were deleted as they had significant number of missing values. The missing values in the remaining columns T_Angle, QRST_Angle and Heart were replaced with the mean value of the corresponding column.

It was observed that the following classes had very few data instances.

![Image4](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image4.PNG)

For model 1, this is not a concern as all the above-mentioned classes will be merged together into a single class named “Arrhythmia”.

As far as building model 2 is concerned, the information available is not sufficient to build a model for predicting the low-instance classes. Moreover, for classes represented by very few instances, training the model by cross-validation will fail as sampling may result in a training dataset that does not contain the information of all the classes. Class 16 was deleted as these were unlabelled classes with no specific pattern and were adding noise to the data. We had initially tested the models with class 16 but no model was able correctly predict this class. Thus, we concluded this class had random unclassified information (as its name suggests) and was not adding much information for the model to learn. Therefore, we removed the instances of this class.

### Merging Classes:

#### Model 1: 
As discussed above, all the instances belonging to classes 2 to 16 were merged to one class.

![Image5](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image5.PNG)

#### Model 2:
The data set is heavily skewed towards class 1 (normal) and hence, the information provided by the attributes of the limited data points of each arrhythmia classes may not be sufficient to individually distinguish them from the normal class. Hence, physiologically similar arrhythmia conditions were identified and merged together. This merging of classes enhances the information available for each class making it easier to separate one from the other.

  1. Old Anterior (class 3) and Old Inferior (class 4) Myocardial Infarction were treated as single class (Myocardial Infarction)
  
  2. Sinus Tachycardy (class 5) and Sinus Bradycardy (class 6) were treated as single class (Sinus Arrhythmia)
  
  3. Left (class 9) and Right (class 10) Bundle branch block will be treated as single class (Bundle Branch Block)

![Image6](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image6.PNG)

### Feature Selection:

Feature selection is the process of selecting a subset of informative and relevant variables which can be used for building the models. A good feature selection helps in reducing the error due to overfitting, simplifies the models and makes them easier to interpret. For model 1, 61 features were selected. Whereas for model 2, 81 variables were selected.
We did not use PCA as it does not consider predictive ability of features. It just selects linear combinations of variables with high variation. It is well suited for data representations in lower dimensions but may not be always desired for predictions. Also, it reduces the interpretability of the features.

For the purpose of feature selection, we used the Boruta library in R.

### Why Boruta?
Boruta algorithm is a wrapper built around the random forest classification algorithm implemented in the R package randomForest. The trees are independently developed on different bagging samples of the training set. The importance measure of an attribute is obtained as the loss of accuracy of classification caused by the random permutation of attribute values between objects. In short, Boruta is based on the same idea which forms the foundation of the random forest classifier, namely, that by adding randomness to the system and collecting results from the ensemble of randomized samples one can reduce the misleading impact of random fluctuations and correlations. In the Boruta package Z score is used as the importance measure because it accounts the fluctuations in mean accuracy loss among the trees.

Boruta follows an all relevant feature search where it heuristically selects a variable which has high relevance with the response and low correlation with other features. It is different from traditional feature selection methods because it does not select a subset which minimizes error in the response by minimal optimal method. Instead it removes the features which are not relevant to the response and divides the relevant features into strongly relevant and weakly relevant based on Z score.

However, we cannot use the Z score calculated in Random Forest as a direct measure of variable importance as this Z score is not directly related to the statistical significance of the variable importance. This makes Boruta well suited for biomedical applications where one might be interested to determine which features are connected in some way to a particular medical condition (target variable).
The feature importance obtained from Boruta is as follows. As can be seen, heart beat rate is the most important variable which makes sense.

![Image7](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image7.PNG)

### Implementation of Models:
Based on the relevant features selected by Boruta package, the models were built using the following algorithms. We have used cross-validation on the training data set to arrive at the best model. Test and train data were split in the ratio of 25:75. Validation set approach was not used as it has high variance.

  1. Logistic regression
  2. LDA
  3. QDA
  4. KNN
  5. Support Vector Machine
  6. Support Vector Classifier
  7. Decision Tree
  8. Random Forest
  9. XgBoost (regularized GBM)
  10. Neural Networks
  
# Implementation Details:
We have applied 10 following statistical learning methods for both the Approaches. Their description, tables, plots and diagnosis is also described for easy understanding and interpretation of the results.

### LDA:
LDA is similar to Bayes classification, however the distribution of data points is considered as normal or Gaussian distribution, which is not the case always. Also the variance of each class is considered the same in LDA classification.

#### Model 1:

As can be seen above, LDA performs poorly in detecting the presence of arrhythmia. The very low sensitivity of the model (0.5806) implies that the model gives the undesirable output of classifying a considerable proportion of arrhythmia patients to normal. Since the model has a high specificity of 0.8039, reducing the threshold for the arrhythmia class can give a better and acceptable trade-off
between sensitivity and specificity.

![Image8](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image8.PNG)

#### Model 2:

Cross-validated LDA model was applied on the multi-class test data set. We can clearly see from the confusion matrix that LDA is able to predict Class 1 and Class 3 with high accuracy, but performs poorly in classifying class 2, 4 & 5. As far as classes with low sensitivity (2, 4 & 5) are concerned, it can be seen from the confusion matrix that majority of misclassified observations have been classified to class 1 (Normal). This is particularly undesirable as this would mean a considerable proportion of the people suffering from Arrhythmia are classified as normal.

Overall AUC of LDA is 0.6799 which means the weighted average of the efficiency over all the classes to rightly classify the observations into their respective classes is only 0.6799, which is very low.

![Image9](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image9.PNG)

#### Note on Poor Performance of LDA:

LDA works on the assumption that the distribution of the predictors in each class is approximately Gaussian. It is also assumed that the predictors share a common variance across all the classes. Since this assumption may not hold true for this data set, LDA may not perform well for detection or classification.

Also, it maybe noted that LDA is a linear classifier. Since we have a large number of features through
boruta package that have a strong relationship with the response, accommodating the effect of all these
features through a linear classifier is difficult.

### Logistic Regression:

Multinomial Logistic Regression is a classification method and is similar to logistic regression. However it is a more generalize model and it is used when there is more than two possible discrete outcomes.

#### Model 1:
Logistic regression model with both ridge and lasso regularization were applied on the training set. Model with higher accuracy was obtained with ridge regularization. This was because we had already selected predictors which had high correlation with the response through Boruta feature selection. This violates the implicit assumption of lasso that some predictors are insignificant to the response variable. Thus, logistic regression with ridge regularization performed better. The optimal value of tuning parameter lambda for ridge regularization was found to be 0.14415.

![Image10](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image10.PNG)

Though the logistic regression does a good as far as accuracy is concerned, 16 arrhythmia cases have been classified as normal, which is undesirable. This can however be improved by reducing the threshold from 0.5 as the objective of the model is only to detect the presence of arrhythmia.

#### Model 2:

Multinomial logistic regression model was applied to the data set to classify the data points into 5 classes. The model gave a very poor test accuracy of 63.46%. As maybe noted from the confusion matrix and test performance measures below, the model performs very poorly in classifying class 4 correctly. The only class the model predicts with a reasonable accuracy is class 1. It may be noted that for multiple classes, discriminant analysis performs better and multinomial logistic regression is not used often. Also, when the number of instances in certain classes is relatively low, logistic regression tends to be unstable.

![Image11](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image11.PNG)

### Support Vector Classifier:

Support vector Classifier uses soft margin to take into account the variance in the training data, whereas in Maximal margin classifier we use hard margin and no violations are allowed.

#### Model 1:
SVM with radial basis function kernel was used to classify the data points. Upon performing grid search on gamma and cost using a 5-fold cross validation on training data set, the optimal parameters were found to be gamma= 0.001953125 and cost=2. The model gave a reasonably high accuracy of 80.53%. However, the sensitivity is relatively low at 0.7581. 207 data points of training data form the support vectors. Since there are multiple classes of arrhythmia with distinct features scattered around in the feature space, a radial separating hyperplane may fail to separate all of them from the distinct ‘normal’ class. Hence, SVM with radial basis function kernel cannot club all the arrhythmia classes and separate them from the single normal class.

![Image12](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image12.PNG)

#### Model 2:

We selected an RBF kernel as the first choice because it can handle cases when the relationship between class labels and attributes is non-linear. This kernel non-linearity maps the samples into a higher dimensional space enabling it to handle the non-linear relationships.

We used radial basis function/ gaussian kernel with gamma= 0.0078125 and cost= 16. We performed a ‘grid search’ on gamma and cost using cross validation. Various values of gamma and cost were tried and the one with the best cross validation accuracy (gamma= 0.0078125 and cost= 16) was picked. We tried over an exponentially growing sequence of cost and gamma where the range of cost was 2-5 to 215 and
that of gamma was from 2-15 to 23. We have used this naïve straightforward grid search approach for cross validation for two reasons. First, computational time required to find good parameters by grid search is not as high as other advanced methods (since there are only two parameters, cost and gamma). Second, grid search can be easily parallelized.

![Image13](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image13.PNG)

We find that SVM with RBF kernel cannot predict class 4 with high sensitivity. Majority of the misclassifications have been classified to class 1 which is particularly undesirable.

We have observed that the cross-validated accuracy is 81% and test accuracy is 71.15%. Since there is a huge difference between the training and test accuracy, it can be concluded that the RBF model is overfitting the data.

It is also observed that there are 223 support vectors in this SVM model which is approximately 71% of the training data. This shows that our model has a low variance and a high bias which is the reason for low test accuracy in comparison to training accuracy.

### K- Nearest Neighbors:
For any positive value of K, the method attempts to find K nearest points to the test data point and assigns the class to it on basis of majority for K nearest points (assigns the value of mean of K nearest points in case of regression).

#### Model 1:
KNN has a very poor sensitivity and very good specificity as classification of normal case is quite accurate KNN uses a small number of neighbors here i.e. 3 and small enough threshold to yield a complex classifier favoring the normal class.

![Image14](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image14.PNG)

#### Model 2:
As seen above, KNN cannot accurately classify test observations for classes 2 and 5 whereas, it completely misclassifies them for class 4. Most of the misclassified observations have been classified to class 1. 

![Image15](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image15.PNG)

KNN is a non-parametric method and makes no assumptions. It classifies a test observation to the class that is most common among its neighbors, which in this case is class 1 since the dataset is imbalanced and skewed towards class 1. Hence, a considerable number of patients who suffer from arrhythmia will not be correctly diagnosed if we follow this model. We find the optimal K that achieves the best accuracy is 7, as seen from the graph, by performing repeated crossvalidation. We use 20 different values to try for each
parameter here. Overall AUC of KNN is 0.7097 which means the weighted average of efficiency over all the classes to rightly classify
observations into their respective classes is only 0.7097.

![Image16](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image16.PNG)

### Support Vector Machine:

Support vector Classifier uses soft margin to take into account the variance in the training data, however Support Vector Machine is used to classify non-linear classes. It does so by expanding the predictor space by use of various type of Kernels.

In our report we have used 3 kernels. 1. Linear 2. Radial and 3. Polynomial

#### Model 1:

SVM with radial basis function kernel was used to classify the data points. Upon performing grid search on gamma and cost using a 5-fold cross validation on training data set, the optimal parameters were found to be gamma= 0.001953125 and cost=2. The model gave a reasonably high accuracy of 80.53%. However, the sensitivity is relatively low at 0.7581. 207 data points of training data form the support vectors. Since there are multiple classes of arrhythmia with distinct features scattered around in the feature space, a
radial separating hyperplane may fail to separate all of them from the distinct ‘normal’ class. Hence, SVM with radial basis function kernel cannot club all the arrhythmia classes and separate them from the single normal class.

![Image17](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image17.PNG)

#### Model 2:
We selected an RBF kernel as the first choice because it can handle cases when the relationship between class labels and attributes is non-linear. This kernel non-linearity maps the samples into a higher dimensional space enabling it to handle the non-linear relationships.

We used radial basis function/ gaussian kernel with gamma= 0.0078125 and cost= 16. We performed a ‘grid search’ on gamma and cost using cross validation. Various values of gamma and cost were tried and the one with the best cross validation accuracy (gamma= 0.0078125 and cost= 16) was picked. We tried over an exponentially growing sequence of cost and gamma where the range of cost was 2-5 to 215 and
that of gamma was from 2-15 to 23. We have used this naïve straightforward grid search approach for cross validation for two reasons. First, computational time required to find good parameters by grid search is not as high as other advanced methods (since there are only two parameters, cost and gamma). Second, grid search can be easily parallelized.

![Image18](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image18.PNG)

We find that SVM with RBF kernel cannot predict class 4 with high sensitivity. Majority of the misclassifications have been classified to class 1 which is particularly undesirable.

We have observed that the cross-validated accuracy is 81% and test accuracy is 71.15%. Since there is a huge difference between the training and test accuracy, it can be concluded that the RBF model is overfitting the data.

It is also observed that there are 223 support vectors in this SVM model which is approximately 71% of the training data. This shows that our model has a low variance and a high bias which is the reason for low test accuracy in comparison to training accuracy.

### Random Forest:
In general, trees which are grown to more depth tend to suffer from over fitting (low bias and high variance). So Random Forest works sin a way similar to bagging of trees, but it only takes a subset of predictors for forming single tree and then take average of all uncorrelated trees so as to reduce variance.

#### Model 1:
Random forest does a very good job in detecting the presence of arrythmia. 10-fold cross validation was used to fit the model. The resulting best model gave a test accuracy of 83.19%. The test performance measures and confusion matrix are as given below. It can be seen that the model has a similar sensitivity and specificity measure.

![Image19](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image19.PNG)

Since the primary goal of this model is to detect the presence of cardiac arrhythmia, a higher sensitivity at the cost of reduced specificity maybe acceptable. A reduced threshold of 0.30 gave the following output.

![Image20](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image20.PNG)

The model increased the sensitivity from 0.8225 to 0.9032 with a small and acceptable reduction in accuracy and specificity.

#### Model 2:
A 5-fold cross validation was performed on training set to obtain a random forest classification model. We cross-validated on number of trees and mtry (number of randomly selected predictors at each split) to find the best model with least cross-validated error. The best model had mtry as 34 and number of trees as 500. We found that Random Forest algorithm is most efficient compared to other models in classifying Arrhythmia into four classes. The cross-validated accuracy of the best model obtained through crossvalidation
was 84.48%.

This is the only classifier that gives a reasonable sensitivity in detecting class 2 and class 4. Random forest classifier also has the overall highest accuracy (Test accuracy of 80.77%). This is because random forest does not suffer from high variance. So even though we have a skewed data set with high number of variables and low number of data points, random forest is able to predict the classes with a reasonably well accuracy. Also, by imparting the randomness in splitting and generating dissimilar trees, random forest tackles the problem of having a single tree which as we saw suffered from the problem of many terminal nodes having class 1 as the majority class.

![Image21](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image21.PNG)

From the graph it can be seen that the accuracy of the model is maximum while considering 34 random predictors at every split.
The final model of random forest has 500 trees at which the error rate stabilizes.

### Decision Trees:
Decision tress are used to create the model that predicts the value of a target variable based on several input variables. It can be used to visually and explicitly represent decision and decision making.

#### Model 1:
The decision tree was fitted on the binary class. The best split is obtained based on Gini index. Upon crossvalidation over the training set to fit the model, it was found that the lowest misclassification error occurred at a tree depth of 9. Hence, the tree was pruned to obtain the sub-tree having 9 terminal nodes. As can be seen from the confusion matrix and test performance measures, the decision tree performs very poorly in identifying the arrhythmia classes.

![Image22](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image22.PNG)

#### Model 2:
Decision trees for multiple class also have a low test accuracy of 69.23% and very low sensitivity for classes 2 and 5. We attribute the low value of accuracy and sensitivity to high variance of the data. The tree classifies the observations based on the majority class at each terminal node. Since the data set is heavily skewed towards class 1, there will be a considerable number of terminal nodes that classify the observation to class 1. This could explain the reason for the observations of class 2-5 being misclassified to class 1.

![Image23](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image23.PNG)

![Image24](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image24.PNG)

This is snippet of the fitted decision tree. Splitting criteria for decision tree is kept as Information gain. We first grew a full decision tree and pruned it based on maximum information gain. We have cross-validated over a range of values of Cp to calculate the best possible spilt. Here Cp value indicates the minimum improvement needed in the model at each node. We have pruned the tree based on the Cp value. The best model has a Cp value of 0.01587302. This value acts a stopping parameter. It speeds up the search for splits because it can identify splits that don’t meet these criteria and prune them before going too far.

### XG-Boost:
Similar to Gradient Boosting in principle, xgboost uses a more regularized model formalization so as to control over-fitting.

#### Model 1:
Extreme gradient boosting does a very good job in detecting the presence of arrythmia. 

![Image25](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image25.PNG)

The area under the curve for binary classification model is 0.8398.

#### Model 2:
We can see from above that extreme gradient boosting classifies the test observations almost correctly with a test accuracy of 84%.
The algorithm applies regularization in addition to boosting by setting parameters like learning rate, subsampling and L2 term on weights to prevent overfitting by reducing variance. We select optimal values of hyperprameters by using a randomized search. Running grid search would be very expensive as we tune seven hyperparameters that govern the tree architecture. There is relatively high specificity and low sensitivity for all classes except class 1 i.e the model correctly identifies those observations that do not belong to these classes. The sensitivity for classes 2 & 3 is low, classes 4 & 5 is moderate while class 1 is high. Sensitivity refers to true positive rate which is highly desirable in cardiac arrhythmia prediction.

![Image26](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image26.PNG)

### Neural Network:
Neural network is a machine learning algorithm that can perform supervised and unsupervised learning tasks. However, in our project, neural network was employed for classification in model 1 and 2. A neural network tries to mimic the functioning of neurons in the human brain. Neural network comprises of an input, output and hidden layers. Each of these layers have processing units referred as neurons. The input(predictors) are fed to the neurons, weights are assigned to the inputs to minimize a predefined loss function, in classification often the misclassification rate. This loss function is minimized using nonlinear optimization models. In this project, library caret, nnet was used in implementing the neural network model. The test dataset was cross-validated for best model selection based on classification accuracy.

#### Model 1:
The optimized hyperparameter values are depth = 7 and decay = 0.7. Depth refers to the number of total layers in the model including input and output layers. Decay is a parameter of the optimization algorithm used in the NN model. Neural network does a decent job in predicting arrhythmic class with a test accuracy of 77.8 and sensitivity of arrhythmic class is 0.725.

![Image27](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image27.PNG)

#### Model 2:
The final values used for the multi-classification model were depth = 7 and decay = 0.5. Here, we get the accuracy of 71.15%. The sensitivity of classes 1, 3 and 4 is good but that of class 2 and 4 is poor. Whereas, the specificity for class 1 is low.

![Image28](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image28.PNG)

## Comparision:

The below Table and Plot shows the accuracy obtained by different approaches used by the team and the improvement gained by used different methods.

### Table:

![Image29](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image29.PNG)

### Plot:

Accuracy V/s Method used (Model-1 and Model-2)

![Image30](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image30.PNG)

## Interpretation:
  1. In general, model 1(model for detection) performs better in terms of accuracy and sensitivity, the critical parameters. This could      be because the initial data set upon which we built the classification model was heavily skewed towards class 1. By merging the          classes, the distribution of classes becomes almost even making it easier to separate the classes with the available information.        Certain classes in model 2(model for classification) have very few instances making it difficult to predict them.
  
  2. All the models classify class 1 (normal) with reasonable accuracy. This could be because more than 50% of the instances belong to        class 1 and hence sufficient information is available for the model to learn its features.
  
  3. Majority of the misclassified instances of class 2 in many models have been classified to class 1(normal) which is highly                undesirable. This could be because there may not be strong features that could distinguish this class from Class 1.
  
  4. We have observed that many cases of class 2 arrhythmia have been misclassified into the normal class. Hence, it may be noted that        there is only a small margin separating the normal class from class 2 arrhythmia.
  
  5. Class 3 has the least number of instances. But almost all the models classify those instances with comparatively high sensitivity.      So it is reasonable to conclude that class 3 (Myocardial Infarction), has very distinct features which make it easier for the model      to identify them.
  
  6. In the classification model (model 2), it is observed that only Random forest and XgBoost are able to classify class 4 with high        accuracy (75% for both). This could be because, unlike the other algorithms, the random forest and Xgboost reduces the variance by      averaging the output from ensemble models.
  
  7. Since, a large number of predictors are used to fit a model on a very small number of observations,the models suffer from high          variance. Random Forest and XgBoost are exceptions.

### Model 1:
In the end, we see that the highest accuracy is 85% for Support Vector Classifier followed by Tree models: Extreme Gradient Boosting Tree and Random Forest (83% and 82% resp.). We get the highest sensitivity of 0.82 for Random Forest which increases to 0.90 by lowering the threshold to 0.3.

![Image31](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image31.PNG)

The ROC curve for the best performing algorithms is shown where:
##### ROC 1: 
    Red <- Logistic Regression with Ridge Regression (AUC=0.8219),
    Green <- Random Forest (AUC=0.8320),
    Blue <- SVM-Radial (AUC=0.8202),
    
##### ROC 2: 
    Black <- Extreme Gradient Boosting Trees (AUC=0.8398)

### Model 2:
Here, the highest accuracy is 84% for Extreme Gradient Boosting Tree and 81% for Random Forest. We average the sensitivity for the arrhythmic classes where the highest value is 0.69 for Extreme Gradient Boosting Tree followed by 0.82 for Random Forest. We also observe that in tree-based models, work well for model 1 and model 2.

## Model Comparision:

![Image32](https://github.com/mustafashabbir10/Cardiac-Arrhythmia/blob/master/Images/Image32.PNG)

# Conclusion:

    1. Through this project, we aimed to detect and classify cardiac arrhythmia into classes. We have
    obtained a very high accuracy with the arrhythmia detection model (model 1). Also, by reducing the
    threshold, the sensitivity of the detection models can be increased further.
    
    2. For the purpose of detection, the models can be utilized as a decision support system and serves the
    objective of the project. However, as far as classification (model 2) to the different classes of
    arrhythmia is concerned, with the given imbalanced data set that is heavily skewed towards class 1,
    obtaining a very accurate classification of all arrhythmia classes is not feasible. But since we have
    identified a set of strong predictors that make sense from a physiological point of view, it can be
    concluded that the models will definitely perform better if trained on a bigger data set that has more
    number of rare class instances.
    
    3. A key result we obtained from the binary classification is that a heart rate less than 51 is a clear
    indication of arrhythmia. So, for patients with a heart rate less than 51, if the model classifies a
    patient to normal class, it could potentially be a case of misclassification and should be looked into.
    
    4. Also, as we expected, personal characteristics like height, age, sex etc. did not make any considerable
    impact.
    
    5. As stated throughout this report, the main intention of this project was to develop an auxiliary
    decision support system to assist medical professionals in detecting and classifying arrhythmia. As far
    as detection is concerned, the model 1 can be used as a decision support system that may guide the
    professionals in confirming the presence of arrhythmia with a fairly high certainty. So as to have an
    efficient classification model (model 2) in place, more data points of the rare classes maybe needed.
    
    6. Since we have utilized only the readings of ECG as the features, adding other features that a medical
    professional may utilize in his diagnosis may improve the performance of the model.






