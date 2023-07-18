# Save and unzip all the Enron 1 to 6 files
# https://www2.aueb.gr/users/ion/data/enron-spam/
# save in a folder named 'enron'

import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_DIR = 'enron'
target_names = ['ham', 'spam']

def get_data(DATA_DIR):
    subfolders = [f'enron{i}' for i in range(1, 7)]
    data = []
    target = []

    for subfolder in subfolders:
        for label in ['spam', 'ham']:
            files = os.listdir(os.path.join(DATA_DIR, subfolder, label))
            for file in files:
                with open(os.path.join(DATA_DIR, subfolder, label, file), encoding='utf-8', errors='ignore') as f:
                    data.append(f.read())
                    target.append(1 if label == 'spam' else 0)

    target = np.array(target)
    return data, target

# Get the data
X, y = get_data(DATA_DIR)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the pipeline
text_clf = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text into a matrix of token counts
    ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation to the token counts
    ('classifier', SGDClassifier(n_jobs=-1))  # Train a linear classifier using the Stochastic Gradient Descent algorithm
])

# Fit the pipeline to the training data
text_clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = text_clf.predict(X_test)

# Print the classification report and accuracy score
print(metrics.classification_report(y_test, y_pred, target_names=target_names))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
