import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Load the dataset (replace 'phishing_dataset.csv' with the path to your dataset)
dataset = pd.read_csv('verified_online.csv')

# Use only the 'url' column for text data
X = dataset['url']
y = dataset['target']

# Preprocess the text data
X = X.apply(preprocess_text)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_vectorized, y_train)

# Predictions for Random Forest
rf_y_pred = rf_classifier.predict(X_test_vectorized)

# Evaluation for Random Forest
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred,average='micro')
rf_recall = recall_score(y_test, rf_y_pred,average='micro')
rf_f1 = f1_score(y_test, rf_y_pred,average='micro')

print("Random Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
# print()

# Initialize and train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_vectorized, y_train)

# Predictions for Decision Tree
dt_y_pred = dt_classifier.predict(X_test_vectorized)

# Evaluation for Decision Tree
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_precision = precision_score(y_test, dt_y_pred,average='micro')
dt_recall = recall_score(y_test, dt_y_pred,average='micro')
dt_f1 = f1_score(y_test, dt_y_pred,average='micro')

print("Decision Tree Metrics:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print()

# Initialize and train Multilayer Perceptron Classifier
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train_vectorized, y_train)

# Predictions for Multilayer Perceptron
mlp_y_pred = mlp_classifier.predict(X_test_vectorized)

# Evaluation for Multilayer Perceptron
mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
mlp_precision = precision_score(y_test, mlp_y_pred,average='micro')
mlp_recall = recall_score(y_test, mlp_y_pred,average='micro')
mlp_f1 = f1_score(y_test, mlp_y_pred,average='micro')

print("Multilayer Perceptron Metrics:")
print("Accuracy:", mlp_accuracy)
print("Precision:", mlp_precision)
print("Recall:", mlp_recall)
print("F1 Score:", mlp_f1)
