import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the dataset
data = pd.read_csv(&quot;D:\Downloads\creditcard.csv&quot;)
# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
# Random under-sampling to balance the dataset
legit_sample = legit.sample(n=492)
ndata = pd.concat([legit_sample, fraud], axis=0)
# Prepare data for training
x = ndata.drop(columns=&#39;Class&#39;, axis=1)
y = ndata[&#39;Class&#39;]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
# Predict on training data and calculate accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(&#39;Accuracy on training data:&#39;, training_data_accuracy)
# Predict on test data and calculate accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(&#39;Accuracy score on test data:&#39;, test_data_accuracy)
