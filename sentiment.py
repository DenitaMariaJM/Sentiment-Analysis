# Sentiment Analysis on Sentiment140 Dataset using NLP

# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Sentiment140 Dataset (Downloaded from Kaggle)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Map sentiment labels: 0 = negative, 4 = positive
df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Display basic information
print(df[['text', 'sentiment']].head())

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing (for performance, sample the dataset)
df = df.sample(n=100000, random_state=42)  # Reduce size for faster processing
df['clean_text'] = df['text'].apply(preprocess_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Vectorize Text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Insights from Model
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_.flatten()
top_positive_coefficients = np.argsort(coefficients)[-10:]
top_negative_coefficients = np.argsort(coefficients)[:10]

plt.figure(figsize=(10, 5))
plt.barh(feature_names[top_positive_coefficients], coefficients[top_positive_coefficients], color='green')
plt.title('Top Positive Words')
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(feature_names[top_negative_coefficients], coefficients[top_negative_coefficients], color='red')
plt.title('Top Negative Words')
plt.show()
