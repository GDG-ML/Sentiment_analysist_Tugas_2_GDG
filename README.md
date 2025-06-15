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
- Matplotlib: for visualization 
- sklearn: for preprocessing and evaluation 
- nltk or keras.preprocessing.text: for text tokenization 
## 📂 Dataset Reference
The dataset used in this project is the **IMDb Movie Reviews Dataset**, which is publicly available.

- **IMDb Dataset Link**: [https://www.kaggle.com/datasets/bhavikjikadara/imdb-dataset-sentiment-analysis/data](https://www.kaggle.com/datasets/bhavikjikadara/imdb-dataset-sentiment-analysis/data)
## 🚀 How to Run the Model
- pip install tensorflow pandas numpy scikit-learn nltk matplotlib seaborn

## 📄 Paper / Research Reference
- https://doi.org/10.1162/neco.1997.9.8.1735
- https://ieeexplore.ieee.org/document/650093
- http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
- https://ieeexplore.ieee.org/abstract/document/6795963
- https://aclanthology.org/P11-1015.pdf
