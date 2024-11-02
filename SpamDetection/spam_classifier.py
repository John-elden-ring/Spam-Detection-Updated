import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

from text_preprocessing import preprocess_text


np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv('spam_ham_dataset.csv')

# Preprocess the text
df['text'] = df['text'].apply(preprocess_text)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label_num'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the neural network
class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Remove sigmoid here; use BCEWithLogitsLoss
        return x



input_dim = X_train.shape[1]
model = SpamClassifier(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensor
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model and vectorizer
torch.save(model.state_dict(), 'spam_classifier.pth')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate accuracy
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)

# Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_outputs = torch.sigmoid(test_outputs)  # Apply sigmoid in evaluation

# Convert outputs to binary predictions (0 or 1)
predicted = (test_outputs >= 0.5).float()

# Calculate accuracy
correct = (predicted.eq(y_test_tensor)).sum().item()
accuracy = correct / y_test_tensor.size(0)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
true_labels = y_test_tensor.cpu().numpy()
predicted_labels = predicted.cpu().numpy()
print(classification_report(true_labels, predicted_labels))
