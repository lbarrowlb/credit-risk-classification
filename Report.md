# Module 20 Report: Credit Risk Classification

## Overview of the Analysis

The analysis was performed using various machine learning techniques to train and evaluate the performance of Logistic Regression Models in identifying the creditworthiness of borrowers. The Logistic Regression Models were trained using different methods to determine the better-performing model comparing their performances. The predictive variables in the model are the labels 0 (healthy loan) and 1 (high-risk loan).

In the process of constructing the models, the dataset was split into features and labels, and further divided into training and testing sets. 
* Machine Learning Model 1 was built by instantiating a logistic regression model and training with the original training sets (X_train, y_train), fitting it to the training sets, and using it to generate predictions. 
* Machine Learning Model 2 was created by resampling the original training data using the RandomOverSampler module, instantiating a logistic regression model and fitting the resampled training sets (X_resample, y_resample) to the model, and generating predictions.

The performance of each model was evaluated based on the balance accuracy score, the confusion matrix, as well as the precision score, recall score, and f1-score in the classification report.

## Results

* Machine Learning Model 1:
  Trained on the original data, Model 1 Accuracy, Precision, and Recall scores are as follows:
  * The accuracy score of 94.4% indicating that it correctly predicted 94.4% of all instances.
  * A precision score of 1.00 for the healthy loan class and 0.87 for the high-risk loan class.
  * The recall score for the healthy loan class is 1.00 and for the high-risk loan class is 0.89. 
  * Overall, the model performs well, with slightly lower precision and recall scores for high-risk loans, but still effective in identifying them.
 
 * Machine Learning Model 2:
  Trained on the resampled data, Model 2 Accuracy, Precision, and Recall scores are as follows:
  * The accuracy score of 99.6% indicating that it correctly predicted 99.6% of all instances.
  * A precision score of 1.00 for the healthy loan class and 0.87 for the high-risk loan class.
  * The recall score for the healthy loan class is 1.00 and for the high-risk loan class is 1.00. 
  Overall, the model performs well, with slightly lower precision score for high-risk loans but high recall scores for both classes, indicating that the model is effective in identifying high-risk loans.

 
## Summary

Based on the analysis, it appears that Model 2 outperforms Model 1 in predicting high-risk loans and has an overall higher accuracy in predicting both labels. Specifically, Model 2 achieved a relatively high precision in predicting high-risk loans while correctly identifying all high-risk loans in the dataset, which is considered a relatively good performance in this context. Therefore, I would recommend using Model 2 in identifying high-risk loans and overall better accuracy in predicting labels.
