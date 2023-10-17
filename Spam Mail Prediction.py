# Import required dependencies 
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset
Mails_dataset = pd.read_csv("/content/mail_data.csv")
# Show the first five columns in the dataset
Mails_dataset.head()
# Show the last five columns in the dataset
Mails_dataset.tail()
# Show the dataset shape
Mails_dataset.shape


# Count the groups in the dataset and plot its repetition
Mails_dataset['Category'].value_counts()
plt.figure(figsize=(5,5))
sns.countplot(x = 'Category',data=Mails_dataset)    


# Check about the none(missing) values in the dataset if will make a data cleaning or not
Mails_dataset.isnull().sum()

# Label the Category column into numeric column by replacing ham value into 1 and spam value into 0
Mails_dataset.replace({'Category':{'ham':1,'spam':0}},inplace=True)



# Split the data into input and label data
X = Mails_dataset['Message']
Y = Mails_dataset['Category']
print(X)
print(Y)
# Make a feature extraction by transforming Message column that is a text data into feature vector to able to be used as input data
feature_extra = TfidfVectorizer(min_df =1 ,stop_words='english', lowercase=True)
X_extracted = feature_extra.fit_transform(X)
# Split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X_extracted,Y,train_size=0.6,random_state=2)
print(X_extracted.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create the model and train it
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)
# Make the model predict on train and test input data
predicted_train_values = LRModel.predict(x_train)
predicted_test_values = LRModel.predict(x_test)
# Avaluate the model
accuracy_predicted_train = accuracy_score(predicted_train_values,y_train)
accuracy_predicted_test = accuracy_score(predicted_test_values,y_test)
print(accuracy_predicted_train)
print(accuracy_predicted_test)



# Build a predictive system
input_Mail = ["Didn't you get hep b immunisation in nigeria",
            "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"]
# Make a feature extraction for the input data
input_Mail= feature_extra.transform(input_Mail)
# Make the model predict the output
if LRModel.predict(input_Mail[0])[0]==1:
    print("the mail is ham")
else:
    print("the mail is spam")
if LRModel.predict(input_Mail[1])[0]==1:
    print("the mail is ham")
else:
    print("the mail is spam")




