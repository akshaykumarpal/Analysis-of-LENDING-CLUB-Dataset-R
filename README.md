# Analysis-of-LENDING-CLUB-Dataset-R

Environment: R (3.3.1) and R Studio (1.0.136)

In this project we had to predict which customers are likely to default in the repayment. As our dataset was leaning towards imbalance, 
we used the "SMOTE" function for oversampling our dataset. Then, we proceeded with six different methods such as Random Forest, Rpart Classification, Naive Bayes, Support Vector Machine, Logistic Linear Regression, ADABoost and evaluated their performances based on different measures such as Confusion matrix, ROC curve and AUC. 
Finally, we came to conclusion that Random Forest model was the best model with high AUC, Accuracy and Recall before and after oversampling to find the borrowers with high chance to default while still keep our good customers.
