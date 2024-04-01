for epoch in range(num_epochs):
    train_running_loss = 0.0
    val_running_loss = 0.0
    train_preds = []
    train_labels = []

    # Training
    full_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = full_model(inputs)
        loss = criterion(outputs, labels)  # Use class indices as target
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_preds.extend(predicted.tolist())
        train_labels.extend(labels.tolist())

    train_loss = train_running_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # Validation
    full_model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        for inputs, labels in val_loader:
            outputs = full_model(inputs)
            loss = criterion(outputs, labels)  # Use class indices as target
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.tolist())
            val_labels.extend(labels.tolist())

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # Update learning rate
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            early_stopped = True
         

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

plt.figure(figsize=(12, 6))

# Plotting training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()