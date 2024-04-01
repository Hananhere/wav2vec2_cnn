# Initialize the simple neural network model
simple_nn = SimpleNN(input_size, hidden_size, output_size)

for param in simple_nn.parameters():
    param.requires_grad = True

# Combine pre-trained model and simple neural network into a single model
full_model = FullModel(model, simple_nn)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

patience = 5
best_val_loss = float('inf')
early_stop_counter = 0
early_stopped = False

max_length = 19000
dataset = EmoDBDataset(root_dir, tokenizer, label_mapping, max_length)

# Define data loaders
batch_size = 32
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import matplotlib.pyplot as plt

# Initialize lists to store training loss and accuracy at each epoch
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []