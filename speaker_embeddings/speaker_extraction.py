#!/usr/bin/env python3
import argparse
import pickle
import torch
import torchaudio
from tqdm import tqdm

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer


def read_wav_scp(wav_scp):
    utt2wav = {}
    with open(wav_scp) as f:
        for line in f:
            utt, wav = line.strip().split(maxsplit=1)
            utt2wav[utt] = wav
    return utt2wav


def main(args):
    device = torch.device(args.device)

    hparams_file = f"{args.exp_dir}/hyperparams.yaml"
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    embedding_model = hparams["embedding_model"].to(device)
    compute_features = hparams["compute_features"].to(device)
    mean_var_norm = hparams["mean_var_norm"].to(device)

    print("Loading checkpoint...")
    checkpointer = Checkpointer(
        checkpoints_dir=args.ckpt,
        recoverables={
            "embedding_model": embedding_model,
            "normalizer": mean_var_norm,
        },
    )
    checkpointer.recover_if_possible()

    embedding_model.eval()
    mean_var_norm.eval()

    utt2wav = read_wav_scp(args.wav_scp)
    print(f"Extracting embeddings for {len(utt2wav)} utterances")

    embeddings = {}

    for utt, wav_path in tqdm(utt2wav.items()):
        signal, sr = torchaudio.load(wav_path)

        if signal.dim() == 2:
            signal = signal.mean(dim=0, keepdim=True)

        signal = signal.to(device)

        with torch.no_grad():
            feats = compute_features(signal)
            feats = mean_var_norm(feats, torch.tensor([feats.shape[1]]).to(device))
            emb = embedding_model(feats)
            emb = emb.squeeze(0).cpu().numpy()

        embeddings[utt] = emb

    print(f"Saving to {args.out_pkl}")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(embeddings, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--exp_dir", required=True,
                        help="exp/asv_anon_anon_B5")
    parser.add_argument("--ckpt", required=True,
                        help="exp/asv_anon_B5/CKPT+10")
    parser.add_argument("--out_pkl", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    main(args)
