import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(rnn_out)
        return out

# Load the trained models and necessary data
@st.cache_resource
def load_models():
    # Load vocabulary
    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Initialize the LSTM model
    lstm_model = LSTMModel(vocab_size=len(vocab), embedding_dim=128, rnn_units=256, dropout_rate=0.02)
    lstm_model.load_state_dict(torch.load('shakespeare_lstm_model.pth', map_location=torch.device('cpu')))
    lstm_model.eval()
    
    # Initialize the RNN model
    rnn_model = RNNModel(vocab_size=len(vocab), embedding_dim=128, rnn_units=256, dropout_rate=0.02)
    rnn_model.load_state_dict(torch.load('rnn_model.pth', map_location=torch.device('cpu')))
    rnn_model.eval()
    
    return lstm_model, rnn_model, word_to_idx, idx_to_word

# Function to generate text
def generate_text(model, start_sequence, word_to_idx, idx_to_word, num_words=5):
    model.eval()
    current_sequence = [word_to_idx.get(word, 0) for word in start_sequence.lower().split()]
    generated_words = []

    for _ in range(num_words):
        with torch.no_grad():
            input_tensor = torch.tensor([current_sequence[-5:]], dtype=torch.long)
            output = model(input_tensor)
            predicted_word_idx = torch.argmax(output, dim=1).item()
        
        predicted_word = idx_to_word[predicted_word_idx]
        generated_words.append(predicted_word)
        current_sequence.append(predicted_word_idx)

    return ' '.join(generated_words)

# Streamlit app
def main():
    st.title("Shakespeare Text Generator")
    st.write("Enter a starting sequence, choose a model, and the AI will generate Shakespeare-like text!")

    lstm_model, rnn_model, word_to_idx, idx_to_word = load_models()

    start_sequence = st.text_input("Enter a starting sequence (at least 5 words):", 
                                   "To be or not to be")
    
    model_choice = st.radio("Choose a model:", ("LSTM", "RNN"))

    if st.button("Generate Text"):
        if len(start_sequence.split()) < 5:
            st.error("Please enter at least 5 words as the starting sequence.")
        else:
            model = lstm_model if model_choice == "LSTM" else rnn_model
            generated_text = generate_text(model, start_sequence, word_to_idx, idx_to_word)
            st.write("Generated Text:")
            st.write(f"{start_sequence} {generated_text}")

if __name__ == "__main__":
    main()