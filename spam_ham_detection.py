# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.patches import FancyBboxPatch

# Load the dataset
file_path = 'spam_ham_dataset.csv'
data = pd.read_csv(file_path, encoding='iso-8859-1')
print(data.head())  # Display the first few rows for debugging

# Visualize the label distribution using pie chart (Figure 3.1: Dataset Overview)
spam_count = data['label'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(spam_count, labels=spam_count.index, autopct='%1.1f%%',
        startangle=90, colors=['lightcoral', 'skyblue'])
plt.title('Proportion of Spam vs Ham Messages')
plt.show()

# Text preprocessing (tokenization, stopword removal, stemming)
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Remove punctuation
    words = [stemmer.stem(word) for word in text.split()
             if word.lower() not in stop_words]
    return ' '.join(words)


data['processed_text'] = data['text'].apply(preprocess_text)

# Prepare the data for the model
X = data['processed_text']
# Convert labels to binary (spam=1, ham=0)
y = pd.get_dummies(data['label'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# ---- Figure 3.2: Preprocessing Pipeline ----


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2))

    steps = ['Raw Text', 'Remove Punctuation',
             'Stopword Removal', 'Stemming', 'Tokenization', 'Padding']
    positions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

    # Draw boxes for each step in the pipeline
    for step, pos in zip(steps, positions):
        ax.add_patch(FancyBboxPatch((pos, 0.5), 0.15, 0.1,
                     boxstyle="round,pad=0.3", edgecolor='black', facecolor='skyblue'))
        plt.text(pos + 0.03, 0.52, step, fontsize=10, ha='center', va='center')

    # Draw arrows between the steps
    for i in range(len(positions) - 1):
        ax.annotate('', xy=(positions[i] + 0.15, 0.55), xytext=(positions[i+1], 0.55),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0.4, 0.8)
    ax.axis('off')

    plt.title("Preprocessing Pipeline", fontsize=12)
    plt.show()


# Call the function to draw the preprocessing pipeline
draw_pipeline()

# ---- End of Preprocessing Pipeline Visualization ----

# Build the hybrid CNN-LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(Conv1D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=64,
                    validation_split=0.2, callbacks=[early_stopping])

# Predictions on the test set
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

# Evaluate the model using performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print classification report for detailed performance
print(classification_report(y_test, y_pred))

# Print individual performance metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize training history: accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Additional visualization for precision, recall, and F1-Score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Performance Metrics')
plt.ylabel('Scores')
plt.show()

# Save the model to a file
model.save('spam_ham_cnn_lstm_model.keras')
