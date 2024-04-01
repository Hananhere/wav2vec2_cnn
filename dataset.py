class EmoDBDataset(Dataset):
    def __init__(self, root_dir, tokenizer, label_mapping, max_length):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_length = max_length
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    file_list.append((file_path, label))
        return file_list

    def _process_audio(self, waveform):
        # Trim or pad the audio to the maximum length
        if waveform.size(1) < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(1)), mode='constant', value=0)
        elif waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        return waveform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]

        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if needed (optional)
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Process the audio
        waveform = self._process_audio(waveform)

        # Tokenize the audio
        input_values = self.tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values.squeeze()

        # Map label to index
        label_index = self.label_mapping[label]

        return input_values, label_index