## 📌 What is this project about? 
This project is a **Sentiment Analysis** deep learning model that classifies text data, such as IMDb movie reviews into positive or negative sentiments. It uses natural language processing (NLP) techniques to clean, tokenize, and preprocess the text data before feeding it into a Bidirectional LSTM neural network.
## What architecture do you use?
We use a **Bidirectional LSTM (Long Short-Term Memory)** neural network architecture implemented using Keras Sequential API. Here's a breakdown of the model layers: 
  
1. **Embedding Layer**  
   Converts each word in the input sequence into a dense vector of fixed size.  
   `Embedding(input_dim=10000, output_dim=64)`

2. **Bidirectional LSTM Layer**  
   Processes the sequence both forward and backward to capture richer context and dependencies.  
   Includes dropout for regularization.  
   `Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))`

3. **Dropout Layers**  
   Added after LSTM and dense layers to reduce overfitting.  
   `Dropout(0.5)`

4. **Dense Hidden Layer**  
   Fully connected layer with ReLU activation to learn complex patterns.  
   `Dense(64, activation='relu')`

5. **Output Layer**  
   A single neuron with sigmoid activation for binary classification (positive or negative sentiment).  
   `Dense(1, activation='sigmoid')`
## What libraries are used?
- TensorFlow / Keras: for model building and training
- NumPy: for numerical operations
- Matplotlib: for visualization (accuracy/loss plots)
- sklearn: for preprocessing and evaluation (e.g., train-test split, accuracy)
- nltk or keras.preprocessing.text: for text tokenization (depending on your setup)
