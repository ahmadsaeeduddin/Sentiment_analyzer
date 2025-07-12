
# Sentiment Analyzer 🚀

This project is a complete NLP pipeline for building, training, evaluating, and saving a **Sentiment Analyzer** using TensorFlow and Keras. It classifies tweets into four categories: **Positive**, **Neutral**, **Negative**, or **Irrelevant**.

---

## 📁 Project Structure

- `SentimentAnalyzer.ipynb`: Jupyter Notebook containing the entire workflow.
- `sentiment_analyzer.keras` and `sentiment_analyzer.h5`: Saved models for re-use.
- `twitter_training.csv` and `twitter_validation.csv`: Training and validation datasets.

---

## 📌 Pipeline Overview

### 1️⃣ **Import Libraries**
```python
import tensorflow as tf
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```
- Use TensorFlow/Keras for deep learning.
- Pandas & NumPy for data manipulation.
- Scikit-learn for splitting the dataset.

### 2️⃣ **Load Dataset**
```python
data = pd.read_csv('twitter_training.csv')
```
- Loads tweets and their sentiments.

### 3️⃣ **Data Cleaning & EDA**
- Rename columns for clarity.
- Check for missing values and duplicates.
- Inspect unique sentiment labels.

### 4️⃣ **Label Encoding**
```python
custom_mapping = {'Positive':0, 'Neutral':1, 'Negative':2, 'Irrelevant':3}
```
- Convert sentiment text labels to numeric for training.

### 5️⃣ **Text Cleaning**
```python
def clean_text(text):
    # Removes URLs, mentions, hashtags, punctuation, and converts to lowercase
```
- Preprocesses tweets to improve model performance.

### 6️⃣ **Tokenization & Vocabulary**
- Custom vocabulary built using word frequency.
- Each tweet is converted to sequences of word indices.
- Sequences are padded to ensure uniform length.

### 7️⃣ **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(...)
```
- 70% training, 30% testing.

### 8️⃣ **Model Architecture**
```python
model = Sequential([
    Embedding(input_dim=50000, output_dim=16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
```
- Embedding layer learns word vectors.
- Global average pooling flattens sequences.
- Dense layers learn complex patterns.
- Softmax for multi-class output.

### 9️⃣ **Training**
```python
EarlyStopping(...)  # Prevents overfitting
history = model.fit(...)
```
- Trained for up to 30 epochs with early stopping.

### 🔟 **Visualization**
- Plots training & validation accuracy and loss to inspect model performance.

### 1️⃣1️⃣ **Interactive Prediction**
```python
predict_sentiment_custom(user_input, vocabulary, model)
```
- Clean user input, tokenize, pad, predict sentiment.

### 1️⃣2️⃣ **Validation**
- Loads `twitter_validation.csv`
- Computes overall accuracy on unseen data.

### 1️⃣3️⃣ **Saving Model**
```python
model.save('sentiment_analyzer.keras')
```
- Saves the trained model for deployment.

---

## ✅ **How to Use**
1. Run the notebook step by step.
2. Train your model with the dataset.
3. Evaluate it on unseen data.
4. Use the interactive input cell to test your own sentences.
5. Save the model and deploy it.

---

## 💡 **Why This Approach?**
- Custom tokenization shows how NLP can be built from scratch.
- Early stopping ensures the model generalizes well.
- Visualization helps understand learning behavior.
- Flexible to adapt to any text classification task.

---
