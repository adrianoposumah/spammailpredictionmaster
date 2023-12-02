"""Importing the Dependencies"""
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Data Collection & Pre-Processing"""
    raw_mail_data = pd.read_csv('C:\\Users\\Acer\\Desktop\\SpamMailPrediction\\app\\mail_data.csv')

    # print(raw_mail_data)

    # replace the null values with a null string
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

    # printing the first 5 rows of the dataframe
    mail_data.head()

    # checking the number of rows and columns in the dataframe
    mail_data.shape

    """Label Encoding"""

    # label spam mail as 0;  ham mail as 1;

    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

    """spam - 0

    ham - 1
    """

    # separating the data as texts and label
    X = mail_data['Message']
    Y = mail_data['Category']

    #print(X)

    #print(Y)

    """Splitting the data into training data & test data"""

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    #print(X.shape)
    #print(X_train.shape)
    #print(X_test.shape)

    """Feature Extraction"""

    # transform the text data to feature vectors that can be used as input to the Logistic regression

    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # convert Y_train and Y_test values as integers

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    #print(X_train)

    #print(X_train_features)

    """Training the Model

    Logistic Regression
    """

    model = LogisticRegression()

    # training the Logistic Regression model with the training data
    model.fit(X_train_features, Y_train)

    """Evaluating the trained model"""

    # prediction on training data

    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    #print('Accuracy on training data : ', accuracy_on_training_data)

    """Building a Predictive System"""

    if request.method == 'POST':
        input_mail = [request.form['email']]
        input_data_features = feature_extraction.transform(input_mail)
        prediction = model.predict(input_data_features)

        if prediction[0] == 1:
            result = 'Ham mail'
        else:
            result = 'Spam mail'

        return render_template('index.html', result=result)

    # If the request method is GET, render the form
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
