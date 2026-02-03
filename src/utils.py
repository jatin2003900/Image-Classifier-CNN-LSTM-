import json, re, torch, numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import nltk
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

class Vocabulary:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.word2id = {PAD:0, BOS:1, EOS:2, UNK:3}
        self.id2word = {0:PAD, 1:BOS, 2:EOS, 3:UNK}

    def build(self, texts):
        tok = lambda s: nltk.word_tokenize(re.sub(r"[^A-Za-z0-9' ]"," ", s.lower()))
        cnt = Counter()
        for t in texts:
            cnt.update(tok(t))
        for w, c in cnt.items():
            if c >= self.min_freq and w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w

    def encode(self, text, max_len=20):
        tok = nltk.word_tokenize(re.sub(r"[^A-Za-z0-9' ]"," ", text.lower()))
        ids = [self.word2id.get(w, self.word2id[UNK]) for w in tok][:max_len]
        return [self.word2id[BOS]] + ids + [self.word2id[EOS]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.id2word.get(int(i), UNK)
            if w in [PAD, BOS]:
                continue
            if w == EOS:
                break
            words.append(w)
        return " ".join(words)

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "word2id": self.word2id}, f)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = cls(min_freq=obj["min_freq"])
        v.word2id = obj["word2id"]
        v.id2word = {int(i):w for w,i in [(i, w) for w,i in v.word2id.items()]}
        return v

class CaptionDataset(Dataset):
    def __init__(self, df, images_root, vocab, split="train", max_len=20, image_size=224):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_root = images_root
        self.vocab = vocab
        self.max_len = max_len
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.images_root}/{row['image_path']}"
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        ids = self.vocab.encode(row["caption"], max_len=self.max_len)
        return img, torch.tensor(ids, dtype=torch.long)

def pad_collate(batch):
    imgs, seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    PAD_ID = 0
    padded = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return imgs, padded, torch.tensor(lengths, dtype=torch.long)

def compute_bleu(gens, refs, n=4):
    import nltk
    weights_map = {
        1:(1.0, 0, 0, 0),
        2:(0.5, 0.5, 0, 0),
        3:(1/3, 1/3, 1/3, 0),
        4:(0.25, 0.25, 0.25, 0.25)
    }
    weights = weights_map.get(n, weights_map[4])
    refs_tok = [[nltk.word_tokenize(r.lower())] for r in refs]
    gens_tok = [nltk.word_tokenize(g.lower()) for g in gens]
    try:
        return nltk.translate.bleu_score.corpus_bleu(refs_tok, gens_tok, weights=weights)
    except ZeroDivisionError:
        return 0.0
