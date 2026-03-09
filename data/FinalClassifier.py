import tensorflow as tf
import numpy as np
import pandas as pd
# Keras Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
# Keras Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import seaborn as sns
# Keras Layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
from tensorflow.keras.layers import Conv1D, MaxPooling1D


acc={}
def createModel():
    df = pd.read_csv('../clean_content.csv')
    # Replace NaN values with empty strings in the 'clean_text' column (or your target column)
    df['clean_text'] = df['clean_text'].fillna('')

    # Ensure all values in the 'clean_text' column are strings
    df['clean_text'] = df['clean_text'].astype(str)

    # split the data
    X = df['clean_text']
    y = df['target']

    # Set the maximum number of words in the vocabulary (5000 most frequent words)
    vocab_size = 10000

    # Set the maximum length for sequences (each text will be padded/truncated to 1000 tokens)
    max_length = 100

    # Set a token to represent out-of-vocabulary (OOV) words during tokenization
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

    tokenizer.fit_on_texts(X)

    X_sequence = tokenizer.texts_to_sequences(X)

    X_padded = pad_sequences(X_sequence, maxlen=max_length, padding='post')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Define embedding dimensions
    embedding_dim = 100  # You can change this value

    # Set sentence length as the max length of the padded sequences
    sentence_length = max_length
    """
    # Build the model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sentence_length),
        LSTM(100),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

    Simple_LSTM = model
    model = Simple_LSTM

    y_log = model.predict(X_test)
    y_pred = np.where(y_log > 0.5, 1, 0)

    acc["Simple LSTM"]=accuracy_score(y_test, y_pred)

    print(accuracy_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))
    
    # Save the model and tokenizer
    Simple_LSTM.save('../models/simple_lstm_model.h5')
    with open('../models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    """
    # Build the model
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sentence_length),
        # 1D Convolutional layer for feature extraction
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        # MaxPooling layer for dimensionality reduction
        MaxPooling1D(pool_size=4),
        # LSTM layer for sequential modeling
        LSTM(100),
        # Dropout for regularization
        Dropout(0.2),
        # Dense layer for binary classification
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Training
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

    CNN_LSTM_Hybrid_model = model
    model = CNN_LSTM_Hybrid_model

    y_log = model.predict(X_test)
    y_pred = np.where(y_log > 0.5, 1, 0)

    acc["CNN_LSTM_Hybrid"]=accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix - CNN lSTM Hybrid Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save as JPG
    plt.savefig(f"../static/vis/cm_cnn_lstm.jpg")
    plt.close()

    # ========== 1. Accuracy vs Epochs ==========
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("../static/vis/acc.jpg")
    plt.close()

    # Save the model
    CNN_LSTM_Hybrid_model.save('../models/CNN_LSTM_Hybrid_model.h5')




#createModel()





