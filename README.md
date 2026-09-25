# 💬 Sentiment Analysis using LSTM

A text sentiment analysis project that uses a **Long Short-Term Memory (LSTM)** deep learning model to classify text into **positive** or **negative** sentiment.

The project includes text preprocessing, tokenization, LSTM model training, model evaluation, and a simple interactive web application built with **Streamlit**.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to identify the emotional polarity of textual data.

In this project, an LSTM-based deep learning model is developed to classify input text into two sentiment categories:

* 😊 **Positive**
* 😞 **Negative**

Users can enter a review, comment, or opinion through the Streamlit interface and receive a predicted sentiment along with its confidence score.

---

## ✨ Features

* Text preprocessing and cleaning
* Tokenization using Keras `Tokenizer`
* Sequence padding with a fixed maximum length
* LSTM-based binary sentiment classification
* Train-test split with reproducible `random_state`
* Early stopping during training
* Model evaluation using accuracy
* Interactive Streamlit web application
* Sentiment prediction with confidence score

---

## 🧠 Model Architecture

The sentiment classifier uses the following neural network architecture:

```text
Input Text
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer (ReLU)
    ↓
Dropout
    ↓
Dense Layer (Sigmoid)
    ↓
Positive / Negative Sentiment
```

### Model Configuration

| Component               | Configuration       |
| ----------------------- | ------------------- |
| Vocabulary Size         | 5,000 words         |
| Maximum Sequence Length | 50                  |
| Embedding Dimension     | 128                 |
| LSTM Units              | 128                 |
| LSTM Dropout            | 0.2                 |
| Recurrent Dropout       | 0.2                 |
| Dense Layer             | 64 units, ReLU      |
| Dropout                 | 0.5                 |
| Output Layer            | 1 unit, Sigmoid     |
| Optimizer               | Adam                |
| Loss Function           | Binary Crossentropy |
| Batch Size              | 32                  |
| Maximum Epochs          | 30                  |
| Early Stopping Patience | 3                   |

The model uses a sigmoid output for binary classification, where predictions greater than or equal to 0.5 are classified as positive and predictions below 0.5 are classified as negative.

---

## 🔄 Text Preprocessing

Before being processed by the model, input text goes through several preprocessing steps:

1. Convert text to lowercase
2. Remove URLs
3. Remove `www` links
4. Remove non-alphabetic characters
5. Normalize consecutive whitespace
6. Tokenize the cleaned text
7. Pad sequences to a maximum length of 50 tokens

The same preprocessing function is used both during model training and when making predictions through the application.

---

## 📊 Dataset

The project uses `data.csv` as the main dataset.

The dataset contains two primary columns:

| Column  | Description                                 |
| ------- | ------------------------------------------- |
| `text`  | Text input used for sentiment analysis      |
| `label` | Sentiment label used as the target variable |

The labels are converted into floating-point values before being used for model training.

The dataset is divided into:

* **80% training data**
* **20% testing data**

using `random_state=42` for reproducibility.

---

## 🏋️ Model Training

The model can be trained using:

```bash
python train_model.py
```

During training, the script:

1. Loads the dataset
2. Cleans the text
3. Creates the target variable
4. Builds the tokenizer
5. Converts text into numerical sequences
6. Applies padding
7. Splits the dataset into training and testing sets
8. Builds the LSTM model
9. Trains the model
10. Evaluates the model on the test set
11. Saves the trained model
12. Saves the tokenizer

The trained model is saved as:

```text
sentiment_lstm.keras
```

while the tokenizer is saved as:

```text
tokenizer.pkl
```

---

## 🌐 Streamlit Application

The project includes an interactive web application built with Streamlit.

Run the application using:

```bash
streamlit run streamlit_app.py
```

The application provides a text input field where users can enter a review, comment, or opinion.

For example:

```text
Pelayanannya sangat baik dan makanannya enak.
```

After clicking **Analyze Sentiment**, the application:

1. Cleans the input text
2. Converts the text into a sequence
3. Pads the sequence
4. Passes it to the trained LSTM model
5. Generates a sentiment prediction
6. Displays the predicted sentiment
7. Displays the model confidence score

---

## 📈 Prediction Output

The application displays:

### Positive Prediction

```text
Sentiment: Positive 😊
Confidence Score: XX.XX%
```

### Negative Prediction

```text
Sentiment: Negative 😞
Confidence Score: XX.XX%
```

The classification threshold is set at **0.5**:

```text
Prediction ≥ 0.5 → Positive
Prediction < 0.5 → Negative
```

---

## 📁 Project Structure

```text
lastproject_textmining/
│
├── data.csv
│
├── text_preprocessing.py
├── train_model.py
├── streamlit_app.py
│
├── sentiment_lstm.keras
├── tokenizer.pkl
│
├── requirements.txt
├── runtime.txt
│
└── .devcontainer/
```

### File Description

| File                    | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `data.csv`              | Dataset used for sentiment classification                        |
| `text_preprocessing.py` | Text cleaning function used by the application                   |
| `train_model.py`        | Script for preprocessing, training, evaluation, and model saving |
| `streamlit_app.py`      | Streamlit web application for sentiment prediction               |
| `sentiment_lstm.keras`  | Trained LSTM model                                               |
| `tokenizer.pkl`         | Saved Keras tokenizer                                            |
| `requirements.txt`      | Python dependencies                                              |
| `runtime.txt`           | Runtime configuration                                            |
| `.devcontainer/`        | Development container configuration                              |

The repository currently contains these project files and model artifacts.

---

## 🛠️ Technologies

This project uses:

* **Python**
* **TensorFlow / Keras**
* **LSTM**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**

The project specifies `tensorflow-cpu==2.16.1` along with Streamlit, NumPy, Pandas, and Scikit-learn in `requirements.txt`.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/asyifaizza/lastproject_textmining.git
cd lastproject_textmining
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option 1 — Use the Existing Trained Model

Since the repository already contains:

```text
sentiment_lstm.keras
tokenizer.pkl
```

you can directly run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL displayed in your terminal.

---

### Option 2 — Train the Model Again

If you want to retrain the model using the dataset:

```bash
python train_model.py
```

After training finishes, the script will generate/update:

```text
sentiment_lstm.keras
tokenizer.pkl
```

Then run:

```bash
streamlit run streamlit_app.py
```

---

## 🔬 Methodology

The overall workflow of this project can be summarized as:

```text
Dataset
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Sequence Padding
   ↓
Train-Test Split
   ↓
LSTM Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Saved Model & Tokenizer
   ↓
Streamlit Application
   ↓
Sentiment Prediction
```

---

## 📊 Evaluation

The model is evaluated using **classification accuracy** on the test dataset.

The evaluation is performed using:

```python
loss, acc = model.evaluate(X_test, y_test)
```

and the resulting accuracy is displayed as a percentage.

> **Note:** The reported accuracy depends on the dataset and training process. For a more comprehensive evaluation, additional metrics such as precision, recall, F1-score, and a confusion matrix can be considered.

---

## ⚠️ Limitations

This project has several limitations:

* The model performs binary sentiment classification only.
* The model is limited to the language and domain represented in the training dataset.
* Text preprocessing focuses on basic cleaning and does not include more advanced linguistic processing.
* Model performance depends heavily on the quality and distribution of the training data.
* The current evaluation primarily uses accuracy.
* The model uses a fixed maximum sequence length of 50 tokens.

Therefore, predictions should be interpreted as model outputs rather than definitive interpretations of sentiment.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate the implementation of **Natural Language Processing and Deep Learning** techniques for sentiment classification, from raw text preprocessing and model development to deployment through an interactive web application.

---

## 👩🏻‍💻 Author

**Asyifa Izza**

Computer Science & Statistics Student

GitHub: [@asyifaizza](https://github.com/asyifaizza)

---

## 📚 Academic Context

This project was developed as part of a **Text Mining / Natural Language Processing** project and demonstrates the application of deep learning, particularly **Long Short-Term Memory (LSTM)**, for text sentiment classification.
