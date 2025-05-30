# Sentiment-Analysis

  This project focuses on sentiment analysis using natural language processing (NLP) techniques on the Sentiment140 dataset, which contains 1.6 million tweets labeled as either positive or negative. The primary objective is to develop a machine learning model that can accurately classify the sentiment expressed in a tweet. The dataset was downloaded from Kaggle and includes tweet text along with corresponding sentiment labels—0 for negative and 4 for positive. For simplicity and computational efficiency, a random sample of 100,000 tweets was used in this analysis. The project uses Python as the core language, supported by libraries such as Pandas and NumPy for data manipulation, NLTK for text preprocessing, scikit-learn for model training and evaluation, and Matplotlib and Seaborn for visualization. 

  The tweets undergo a thorough preprocessing phase where URLs, punctuation, numbers, and HTML tags are removed. The text is then lowercased, tokenized, lemmatized, and stripped of common stopwords to retain only meaningful words. The cleaned text is converted into numerical features using the Term Frequency-Inverse Document Frequency vectorizer dimensionality reduction. The dataset is split into training and testing subsets using an 80-20 split. A logistic regression model, a widely used and efficient classifier for binary problems, is trained on the TF-IDF-transformed training data. Once trained, the model predicts the sentiment labels for the test data. Its performance is evaluated using a classification report that includes metrics like precision, recall, and F1-score. To better understand the model’s accuracy, a confusion matrix is generated and visualized as a heatmap. The confusion matrix displays true positives, true negatives, false positives, and false negatives, which help assess where the model performs well and where it makes errors. For instance, true positives represent tweets correctly identified as positive, while false negatives indicate tweets that were actually positive but misclassified as negative.
  
  In addition to evaluation metrics, the model’s learned coefficients are used to identify the top words that strongly influence sentiment classification. Positive coefficients represent words likely to appear in positive tweets (e.g., “love,” “great”), while negative coefficients correspond to words frequently found in negative tweets (e.g., “hate,” “worst”). These top words are visualized using horizontal bar charts to provide insights into what drives sentiment in social media text.
  


#OUTPUT

![Image](https://github.com/user-attachments/assets/83ab17f1-232a-445e-92b0-673d0015b9d4)

![Image](https://github.com/user-attachments/assets/e9815a47-432f-4612-9dd6-78a9ccd04122)
