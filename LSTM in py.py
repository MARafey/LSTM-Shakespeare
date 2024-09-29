import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
import spacy
import string
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load NLTK's stop words list and set of punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

# Step 1: Load and preprocess the text data using NLTK
def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read().lower()  # Convert to lowercase
    
    # Split the text into chunks of 500,000 characters
    chunks = [text[i:i + 500000] for i in range(0, len(text), 500000)]
    
    tokens = []
    for chunk in chunks:
        # Tokenization using NLTK
        chunk_tokens = word_tokenize(chunk)
        
        # Stopword and punctuation removal
        filtered_tokens = [token for token in chunk_tokens if token not in stop_words and token not in punctuations]
        
        tokens.extend(filtered_tokens)
    
    return tokens

# Load and preprocess the data
tokens = load_and_preprocess_data('Data/alllines.txt')


# Step 2: Create a vocabulary and convert tokens to integer sequences
def create_sequences(tokens, seq_length):
    # Create vocabulary
    vocab = sorted(set(tokens))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Convert tokens to integers
    text_as_int = np.array([word_to_idx[word] for word in tokens])
    
    # Prepare overlapping sequences
    sequences = []
    for i in range(seq_length, len(text_as_int)):
        sequences.append(text_as_int[i-seq_length:i+1])  # Input: seq_length words, Target: next word
    
    return np.array(sequences), vocab, word_to_idx

# Create sequences
seq_length = 5
sequences, vocab, word_to_idx = create_sequences(tokens, seq_length)
x, y = sequences[:, :-1], sequences[:, -1]  # Split into input and target

# Convert data to PyTorch tensors
x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Step 3: Define the LSTM model
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
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Use only the last output
        out = self.fc(lstm_out)
        return out


        from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
import torch.optim as optim

# Step 4: Define extensive grid of hyperparameters
param_grid = {
    'embedding_dim': [128, 256],
    'rnn_units': [256, 512],
    'dropout_rate': [0.2, 0.5],
}

# Track the loss for each hyperparameter configuration
results = []
best_score = float('inf')
best_params = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform manual grid search
for params in ParameterGrid(param_grid):
    print(f"Training with params: {params}")

    # Initialize the model
    model = LSTMModel(vocab_size=len(vocab), embedding_dim=params['embedding_dim'], 
                      rnn_units=params['rnn_units'], dropout_rate=params['dropout_rate']).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 16  # Adjust the batch size here

    # Train the model
    model.train()
    losses = []
    for epoch in range(10):
        epoch_loss = 0.0
        num_batches = len(x) // batch_size
        for i in range(0, len(x), batch_size):
            # Get mini-batch data and move to device
            batch_x = x[i:i+batch_size].to(device)
            batch_y = y[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/10, Loss: {avg_epoch_loss}")
        losses.append(avg_epoch_loss)  # Record loss for this epoch

    # Evaluate model
    model.eval()  # Switch to evaluation mode
    eval_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            batch_y = y[i:i+batch_size].to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            eval_loss += loss.item()

    avg_eval_loss = eval_loss / num_batches
    print(f"Final Score (loss): {avg_eval_loss}")

    # Store the results for comparison
    results.append({
        'params': params,
        'losses': losses,
        'final_loss': avg_eval_loss
    })

    if avg_eval_loss < best_score:  # Compare loss values
        best_score = avg_eval_loss
        best_params = params

    # Step 6: Retrain the model with the best hyperparameters and save it
# final_model = LSTMModel(vocab_size=len(vocab), embedding_dim=best_params['embedding_dim'],
#                         rnn_units=best_params['rnn_units'], dropout_rate=best_params['dropout_rate'])

final_model = LSTMModel(vocab_size=len(vocab), embedding_dim=128,
                        rnn_units=256, dropout_rate=0.02).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
batch_size = 16
# Retrain with more epochs
final_model.train()

for epoch in range(10):
    epoch_loss = 0.0
    num_batches = len(x) // batch_size
    for i in range(0, len(x), batch_size):
        # Get mini-batch data and move to device
        batch_x = x[i:i+batch_size].to(device)
        batch_y = y[i:i+batch_size].to(device)

        optimizer.zero_grad()
        output = final_model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

            # Accumulate the loss
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/10, Loss: {avg_epoch_loss}")
        
        # Save the model
torch.save(final_model.state_dict(), 'shakespeare_lstm_model.pth')


def generate_text(model, start_string, num_generate=5, temperature=1.0):
    # Evaluation mode
    model.eval()

    # Tokenize the starting string
    start_tokens = [word_to_idx[word] for word in start_string if word in word_to_idx]
    input_eval = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize the generated text
    text_generated = start_string

    # Generate words
    with torch.no_grad():
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions.squeeze(0) / temperature
            predicted_id = torch.multinomial(torch.exp(predictions), 1).item()

            # Convert the predicted token to a word
            predicted_word = vocab[predicted_id]

            # Update the generated text
            text_generated += " " + predicted_word

            # Update the input for the next word
            input_eval = torch.tensor([predicted_id], dtype=torch.long).unsqueeze(0).to(device)

    return text_generated