import argparse, os, json, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import EncoderCNN, DecoderLSTM
from utils import Vocabulary, CaptionDataset, pad_collate, compute_bleu

def plot_curves(history, outpath):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (CE)"); ax.set_title("Training & Validation Loss")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def plot_bleu(bleus, outpath):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(bleus, label="BLEU-4")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BLEU"); ax.set_title("Validation BLEU-4")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="CSV with columns: image_path, caption, split")
    ap.add_argument("--images-root", type=str, default="data")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.captions)
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(df[df["split"]=="train"]["caption"].tolist())
    vocab.to_json(os.path.join(args.outdir, "vocab.json"))

    train_ds = CaptionDataset(df, args.images_root, vocab, split="train", max_len=args.max_len)
    val_ds = CaptionDataset(df, args.images_root, vocab, split="val", max_len=args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = EncoderCNN(embed_dim=args.embed_dim).to(device)
    dec = DecoderLSTM(vocab_size=len(vocab.word2id), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)

    crit = nn.CrossEntropyLoss(ignore_index=0)
    params = list(dec.parameters()) + list(enc.fc.parameters()) + list(enc.bn.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "bleu4": []}
    best_bleu = -1.0
    for ep in range(1, args.epochs+1):
        enc.train(); dec.train(); tr_loss = 0.0; n=0
        for imgs, tgt, lengths in tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs} [train]"):
            imgs, tgt = imgs.to(device), tgt.to(device)
            opt.zero_grad()
            feats = enc(imgs)
            logits = dec(feats, tgt[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward(); opt.step()
            tr_loss += loss.item() * imgs.size(0); n += imgs.size(0)
        tr = tr_loss / n

        enc.eval(); dec.eval(); va_loss=0.0; vn=0
        gens, refs = [], []
        with torch.no_grad():
            for imgs, tgt, lengths in tqdm(val_dl, desc=f"Epoch {ep}/{args.epochs} [val]"):
                imgs, tgt = imgs.to(device), tgt.to(device)
                feats = enc(imgs)
                logits = dec(feats, tgt[:, :-1])
                loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                va_loss += loss.item() * imgs.size(0); vn += imgs.size(0)
                out_ids = dec.sample(feats, max_len=args.max_len)
                for i in range(out_ids.size(0)):
                    gens.append(vocab.decode(out_ids[i].cpu().numpy()))
                    refs.append(vocab.decode(tgt[i, 1:].cpu().numpy()))
        vl = va_loss / vn
        bleu4 = compute_bleu(gens, refs, n=4)

        history["train_loss"].append(tr); history["val_loss"].append(vl); history["bleu4"].append(bleu4)
        print(f"[epoch {ep}] train_loss={tr:.4f} val_loss={vl:.4f} bleu4={bleu4:.4f}")
        if bleu4 > best_bleu:
            best_bleu = bleu4
            torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                        "vocab_size": len(vocab.word2id), "embed_dim": args.embed_dim,
                        "hidden_dim": args.hidden_dim, "max_len": args.max_len},
                       os.path.join(args.outdir, "best_captioner.pt"))

        plot_curves(history, os.path.join(args.outdir, "training_curves.png"))
        plot_bleu(history["bleu4"], os.path.join(args.outdir, "bleu_scores.png"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"best_bleu4": best_bleu}, f, indent=2)
    print("[OK] Training done. Best BLEU-4:", best_bleu)

if __name__ == "__main__":
    main()
