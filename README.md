# Spam-Mail-Prediction
This is a Spam Mail prediction model that uses a logistic regression algorithm to classify emails as either spam or ham (non-spam).

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
-Pandas
-Matplotlib
-Seaborn
-Scikit-learn
-Numpy

## Overview of the Code
1-Load the Spam Mail dataset and display the first and last five rows.

2-Data Exploration:
- Count the number of emails in each category (spam or ham) and visualize it.

3-Data Cleaning:
- Check for missing values in the dataset.
- Label encode the 'Category' column by replacing 'ham' with 1 and 'spam' with 0.

4-Split the dataset into input features (X) and labels (Y).

5-Perform feature extraction by transforming the text data in the 'Message' column into a feature vector using the TfidfVectorizer.

6-Split the data into training and testing sets.

7-Create a logistic regression model, train it on the training data, and predict both the training and test data.

8-Evaluate the model's accuracy on the training and test data.

9-Build a predictive system to classify emails as spam or ham for new input data.


## Model Accuracy
The model has achieved an accuracy of 95% on the test data.


## Contribution
Contributions to this project are welcome. 
You can help improve the accuracy of the model, explore more advanced NLP techniques for feature extraction, or enhance the data preprocessing and visualization steps. 
Please feel free to make any contributions and submit pull requests.

