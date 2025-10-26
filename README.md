# Movie-Sentiment-Analyzer
Developed a movie review sentiment analyzer using Python. Implemented TF-IDF, Word2Vec, Random Forest, and LSTM models on 50,000 reviews. Achieved high accuracy predicting positive/negative sentiment through deep learning and feature engineering.

Key Features
Dataset: 50,000 labeled IMDB movie reviews (positive/negative sentiment).

Text Preprocessing: Lowercasing, punctuation/stops removal, lemmatization for better feature extraction.

Feature Engineering:

TF-IDF vectorization for classic machine learning.

Word2Vec embeddings for semantic vector-based modeling.

Tokenization and padding for sequential models.

Models Implemented:

Logistic Regression with TF-IDF features.

Random Forest with Word2Vec averaged vectors.

LSTM Deep Neural Network for sequence modeling, handling word order and context.

Regularization and Overfitting Control: Dropout layers and validation monitoring in LSTM.

Model Evaluation: Accuracy, confusion matrix, precision, recall, and train/validation learning curves.

Tech Stack: Python, Scikit-learn, Gensim, TensorFlow/Keras, Pandas, Matplotlib, NLTK/Spacy.

Results
Achieved high accuracy and robust generalization for real-world movie review sentiment analysis.

Compared feature representations and algorithms for best performance.

All code modularized for easy experimentation and extension.
