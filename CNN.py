class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNN, self).__init__()
    self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.fc = nn.Linear(hidden_size, output_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.mean(dim=2) # Global average pooling
        x = self.fc(x)
        x = self.softmax(x)
        return x