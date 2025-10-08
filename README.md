
# Spam Classifier 🚫📩

A machine learning project that classifies SMS messages as **Spam** or **Ham** (not spam) using **TF-IDF vectorization** and **Multinomial Naive Bayes**. This project demonstrates an end-to-end NLP workflow, including text preprocessing, model training, evaluation, and deployment via a **Gradio web app**.

---

## 🧠 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn, Gradio  
- **Model:** Multinomial Naive Bayes  
- **Feature Extraction:** TF-IDF Vectorization (unigrams and bigrams)  
- **UI / Deployment:** Gradio  

---

## 📊 Dataset
- **Source:** [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)  
- **Size:** 5,572 SMS messages  
- **Labels:**  
  - `ham = 0` → Not Spam  
  - `spam = 1` → Spam  
- **Split:** 80% train, 20% test  

---

## 🏗️ Project Pipeline

1. **Data Setup & Exploration**  
   - Load dataset and inspect columns  
   - Map labels to 0 (ham) and 1 (spam)  

2. **Text Preprocessing**  
   - Convert text to lowercase  
   - Remove punctuation, numbers, and stopwords  
   - Optionally lemmatize words  

3. **Feature Extraction**  
   - Convert text into numerical features using **TF-IDF Vectorization**  
   - Use unigrams and bigrams for better phrase detection  

4. **Model Training & Evaluation**  
   - Train **Multinomial Naive Bayes** on TF-IDF features  
   - Evaluate using **accuracy**, **classification report**, and **confusion matrix**  
   - Accuracy achieved: ~95%  

5. **Deployment (Gradio Web App)**  
   - Input an SMS message to classify as **Spam 🚫** or **Ham ✅**  
   - Quick and interactive interface for testing  

---

## 📈 Model Performance

- **Accuracy:** ~95% on test set  
- **Classification Report:** Included in the notebook  
- **Confusion Matrix:** Included in the notebook  

---

## 🚀 Usage Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/UswahAA/spam-classifier.git
cd spam-classifier
````

2. **Open the notebook and run all cells** to train or test the model.

3. **Launch Gradio app** to test messages interactively:

```python
import gradio as gr
import joblib

# Load model & vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "🚫 Spam" if pred == 1 else "✅ Ham"

gr.Interface(fn=predict_spam, inputs="text", outputs="text", title="Spam Classifier").launch()
```

4. **Test sample messages**:

* `Congratulations! You won a $1000 gift card.` → 🚫 Spam
* `Hey, are we meeting for lunch tomorrow?` → ✅ Ham

---

## 📂 Saved Files

* `spam_classifier_model.pkl` → Trained Naive Bayes model
* `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer
* `spam.csv` → Original dataset

---


## ⚠️ Note

This model is trained on a **static dataset**. Real-world messages may vary, so predictions may not always be perfect.

---

## ✨ Author

**Uswah AA**
[GitHub](https://github.com/UswahAA)


