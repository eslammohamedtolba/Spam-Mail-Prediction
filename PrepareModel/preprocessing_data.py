# Import required dependencies 
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk


def load_ds(path = "mail_data.csv"):
    Mails_dataset = pd.read_csv(path)
    return Mails_dataset

def data_analysis(Mails_dataset):
    # Show the first five columns in the dataset
    print("the first five instances are",Mails_dataset.head())
    # Show the last five columns in the dataset
    print("the last five instances are",Mails_dataset.tail())
    # Show the dataset shape
    print("the dataset shape",Mails_dataset.shape)
    # Count the groups in the dataset and plot its repetition
    print(Mails_dataset['Category'].value_counts())
    # Check about the none(missing) values in the dataset if will make a data cleaning or not
    print("number of null values is",Mails_dataset.isnull().sum().sum())
    plt.figure(figsize=(5,5))
    sns.countplot(x = 'Category',data=Mails_dataset)    
    plt.show()

def data_preproc(Mails_dataset):
    # Label the Category column into numeric column by replacing ham value into 1 and spam value into 0
    Mails_dataset.replace({'Category':{'ham':1,'spam':0}},inplace=True)
    return Mails_dataset

def split_data(Mails_dataset):
    # Split the data into input and label data
    X = Mails_dataset['Message']
    Y = Mails_dataset['Category']
    print(X[:5])
    print(Y[:5])
    return X,Y

def feature_extraction(X):
    feature_extra = TfidfVectorizer(min_df =1 ,stop_words='english', lowercase=True)
    X_extracted = feature_extra.fit_transform(X)
    # Save the fitted vectorizer with pickle
    with open('tfidf_vectorizer.pk', 'wb') as f:
        pk.dump(feature_extra, f)
    return X_extracted

def split_train_test_data(X,Y):
    # Split data into train and test data
    x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.6,random_state=2)
    # print(shape of train and test data)
    print(X.shape,x_train.shape,x_test.shape)
    print(Y.shape,y_train.shape,y_test.shape)
    return x_train,x_test,y_train,y_test



