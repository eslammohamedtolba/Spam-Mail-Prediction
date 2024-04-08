from preprocessing_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle as pk

def CreateTrainAvaluate_model(x_train, x_test ,y_train, y_test):
    # Create the model and train it
    LRModel = LogisticRegression()
    LRModel.fit(x_train,y_train)
    # Make the model predict on train and test input data
    predicted_train_values = LRModel.predict(x_train)
    predicted_test_values = LRModel.predict(x_test)
    # Avaluate the model
    accuracy_predicted_train = accuracy_score(predicted_train_values,y_train)
    accuracy_predicted_test = accuracy_score(predicted_test_values,y_test)
    return LRModel, accuracy_predicted_train, accuracy_predicted_test


# Load the dataset
Mails_dataset = load_ds()
# analyse the dataset
data_analysis(Mails_dataset)
# preprocess the dataset
Mails_dataset = data_preproc(Mails_dataset)
# Split the data into input and label data
X,Y = split_data(Mails_dataset)
# Make a feature extraction by transforming Message column that is a text data into feature vector to able to be used as input data
X_extracted = feature_extraction(X)
# Split data into train and test data
x_train, x_test, y_train, y_test =  split_train_test_data(X_extracted,Y)


# create, train and avaluate_model
LRModel, acc_on_train, acc_on_test = CreateTrainAvaluate_model(x_train, x_test ,y_train, y_test) 
# Show the accuracy of the model on train and test data
print(acc_on_train)
print(acc_on_test)

# Save model
pk.dump(LRModel , open('LRModel.sav','wb'))


