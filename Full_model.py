input_size = 768  # Size of features extracted from pre-trained model
hidden_size = 256
output_size = num_classes  # Number of emotion classes
learning_rate = 0.001
num_epochs = 25
batch_size = 32
root_dir = "/content/drive/MyDrive/BTP_hanan_dataset/Dataset/EmoDB"

class FullModel(nn.Module):
    def __init__(self, wav2vec_model, simple_nn_model):
        super(FullModel, self).__init__()
        self.wav2vec_model = wav2vec_model
        self.simple_nn_model = simple_nn_model

    def forward(self, x):
        # Get hidden states from pre-trained model
        hidden_states = self.wav2vec_model(x)[0]

        # Aggregate hidden states (e.g., by averaging or max-pooling)
        hidden_states = hidden_states.permute(0, 2, 1)
  # Example: averaging
        
        # Pass through simple neural network
        output = self.simple_nn_model(hidden_states)

        return output