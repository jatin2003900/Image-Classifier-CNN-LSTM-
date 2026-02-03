import torch, torch.nn as nn
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in backbone.parameters():
            p.requires_grad = False
        modules = list(backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            feats = self.cnn(images).squeeze()
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
        feats = self.fc(feats)
        feats = self.bn(feats)
        return torch.relu(feats)

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        emb = self.embed(captions)
        feats = features.unsqueeze(1)
        x = torch.cat([feats, emb], dim=1)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

    def sample(self, features, max_len=20, bos_id=1, eos_id=2):
        B = features.size(0)
        inputs = features.unsqueeze(1)
        states = None
        outputs = []
        for _ in range(max_len):
            out, states = self.lstm(inputs, states)
            logits = self.fc(out[:, -1, :])
            _, next_ids = torch.max(logits, dim=1)
            outputs.append(next_ids)
            emb = self.embed(next_ids).unsqueeze(1)
            inputs = emb
            if (next_ids == eos_id).all():
                break
        return torch.stack(outputs, dim=1)
