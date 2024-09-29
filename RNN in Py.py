import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re

# Preprocessing function
def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read().lower()  # Convert to lowercase
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = text.split()
    return tokens

# Create sequences from tokens
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

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc(x)
        return x

# Load and preprocess the data
tokens = load_and_preprocess_data('/content/alllines.txt')

# Create sequences
seq_length = 5
sequences, vocab, word_to_idx = create_sequences(tokens, seq_length)
x, y = sequences[:, :-1], sequences[:, -1]  # Split into input and target

# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Set hyperparameters directly
embedding_dim = 256
rnn_units = 512
dropout_rate = 0.2
batch_size = 16
num_epochs = 10

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(vocab_size=len(vocab), embedding_dim=embedding_dim, 
                 rnn_units=rnn_units, dropout_rate=dropout_rate).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
losses = []
for epoch in range(num_epochs):
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
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

# Save the model
torch.save(model.state_dict(), 'rnn_model.pth')
print("Model saved as 'rnn_model.pth'")

def generate_text(model, start_string, num_generate=5):
    input_eval = torch.tensor([[word_to_idx[word] for word in start_string.split()]], dtype=torch.long).to(device)
    text_generated = start_string

    model.eval()
    with torch.no_grad():
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = torch.softmax(predictions, dim=-1)
            predicted_id = torch.multinomial(predictions, num_samples=1)
            predicted_word = vocab[predicted_id.item()]
            text_generated += ' ' + predicted_word
            input_eval = torch.tensor([[predicted_id.item()]], dtype=torch.long).to(device)

    return text_generated