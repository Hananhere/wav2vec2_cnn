tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Modify the last layer to match the number of output classes
label_mapping = {'anger W': 0, 'fear A': 1, 'happiness F': 2, 'neutral N': 3}
num_classes = len(label_mapping)

model.lm_head = nn.Linear(in_features=model.config.hidden_size, out_features=num_classes, bias=True)

for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True
