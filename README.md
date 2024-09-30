
# Spam vs Ham Detection Using Hybrid CNN-LSTM Model

## Overview

This repository contains a Python-based project for detecting spam vs ham messages using a hybrid CNN-LSTM deep learning model. The project preprocesses textual data from a dataset, builds a hybrid model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM), and evaluates its performance in classifying messages as spam or ham.

## Features

- Text Preprocessing: Tokenization, stopword removal, stemming, and padding for uniform sequence lengths.
- Hybrid CNN-LSTM Model: Combines CNN for feature extraction and LSTM for sequence modeling to classify text data.
- Visualization: Includes pie chart for data distribution, confusion matrix, and accuracy/validation accuracy plots.
- Performance Metrics: Calculates accuracy, precision, recall, and F1-Score for model evaluation.
- Confusion Matrix: Visual representation of model predictions.
- Early Stopping: Prevents overfitting during model training by stopping when validation performance stops improving.

## Installation

### Prerequisites
Ensure that you have Python 3.x installed on your machine. You will also need the following libraries:

- Pandas
- NumPy
- NLTK
- Scikit-Learn
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Wordcloud

To install the required libraries, run:

```bash
pip install pandas numpy nltk scikit-learn tensorflow matplotlib seaborn
```

### Steps
1. Clone the repository:
```bash
git clone https://github.com/masood2004/spam_ham_detection.git
cd spam-ham-detection
```

2. Download the dataset and save it as spam_ham_dataset.csv.

3. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

4. Run the program:

```bash
python spam_ham_detection.py
```

## Usage

- Data Preprocessing: The program preprocesses text data by removing punctuation, stopword removal, stemming, tokenizing, and padding the text for the CNN-LSTM model.

- Model Training: The hybrid model (CNN-LSTM) is trained on the preprocessed text data to classify messages as spam or ham.

- Evaluation: Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix are generated to evaluate the model.

- Visualization: Plots showing accuracy, confusion matrix, and key performance metrics are generated for analysis.

- Saving Model: The trained model is saved in the .keras format.

## Example

To train and evaluate the CNN-LSTM model for spam vs ham detection:

```bash
python spam_ham_detection.py
```

This command will load the dataset, preprocess the text, train the hybrid model, and display evaluation metrics.

## Visualizations

- Dataset Overview: A pie chart visualizing the proportion of spam vs ham messages.
- Preprocessing Pipeline: Visualization of the text preprocessing pipeline (e.g., removing punctuation, tokenization, etc.).
- Confusion Matrix: A heatmap showing the confusion matrix for spam vs ham classification.
- Training History: Accuracy and validation accuracy over training epochs.


## Dependencies

- Pandas: For data manipulation and analysis.
- NumPy: For numerical computations.
- NLTK: For text preprocessing (tokenization, stopword removal, etc.).
- Scikit-Learn: For splitting the dataset and evaluating model performance.
- TensorFlow/Keras: For building and training the hybrid CNN-LSTM model.
- Matplotlib & Seaborn: For data visualizations and plotting confusion matrix.

To install all dependencies:

```bash
pip install -r requirements.txt
```

## Performance Metrics

- Accuracy: Measures how often the classifier makes the correct prediction.
- Precision: Measures the percentage of relevant instances among retrieved instances.
- Recall: Measures the percentage of relevant instances that were retrieved.
- F1-Score: Harmonic mean of precision and recall.

## Contribution

Contributions are welcome! To contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature-branch).
- Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For questions or support, feel free to contact:

Name: Syed Masood Hussain
Email: hmasood3288@gmail.com
GitHub: masood2004
