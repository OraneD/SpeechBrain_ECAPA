#!/usr/bin/env python3
"""
Extract speaker embeddings from ECAPA-TDNN model trained on LibriSpeech.

Usage:
    python extract_embeddings.py --wav_scp <path> --checkpoint <path> --output <path> [--config <path>]
"""

import os
import sys
import argparse
import pickle
import torch
import torchaudio
import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import numpy as np


def read_wav_scp(wav_scp_path):
    """
    Read Kaldi-style wav.scp file.
    
    Format: utt_id /path/to/audio.wav
    or: utt_id sox /path/to/audio.wav -t wav - |
    
    Returns dict: {utt_id: wav_path}
    """
    wav_dict = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            utt_id, wav_path = parts
            if '|' in wav_path:
                print(f"Warning: Pipe commands not supported, skipping {utt_id}")
                continue
            wav_dict[utt_id] = wav_path
    return wav_dict


class EmbeddingExtractor:
    """Extract embeddings using trained ECAPA-TDNN model."""
    
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Initialize the embedding extractor.
        
        Args:
            checkpoint_path: Path to the checkpoint directory (CKPT+XX/) or .ckpt file
            config_path: Path to hyperparams yaml (optional, uses default if None)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        if config_path is None:
            self.hparams = self._get_default_hparams()
        else:
            with open(config_path) as f:
                self.hparams = load_hyperpyyaml(f)
        
        self.compute_features = Fbank(
            n_mels=self.hparams.get('n_mels', 80),
            left_frames=self.hparams.get('left_frames', 0),
            right_frames=self.hparams.get('right_frames', 0),
            deltas=self.hparams.get('deltas', False)
        ).to(device)
        
        self.mean_var_norm = InputNormalization(
            norm_type='sentence',
            std_norm=False
        ).to(device)
        
        self.embedding_model = ECAPA_TDNN(
            input_size=self.hparams.get('n_mels', 80),
            channels=self.hparams.get('channels', [512, 512, 512, 512, 1536]),
            kernel_sizes=self.hparams.get('kernel_sizes', [5, 3, 3, 3, 1]),
            dilations=self.hparams.get('dilations', [1, 2, 3, 4, 1]),
            attention_channels=self.hparams.get('attention_channels', 128),
            lin_neurons=self.hparams.get('lin_neurons', 192)
        ).to(device)
        
        self._load_checkpoint(checkpoint_path)
        
        self.compute_features.eval()
        self.mean_var_norm.eval()
        self.embedding_model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {device}")
    
    def _get_default_hparams(self):
        """Default hyperparameters matching training configuration."""
        return {
            'n_mels': 80,
            'left_frames': 0,
            'right_frames': 0,
            'deltas': False,
            'channels': [512, 512, 512, 512, 1536],
            'kernel_sizes': [5, 3, 3, 3, 1],
            'dilations': [1, 2, 3, 4, 1],
            'attention_channels': 128,
            'lin_neurons': 192,
        }
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if os.path.isdir(checkpoint_path):
            print(f"Loading SpeechBrain checkpoint directory: {checkpoint_path}")
            
            emb_ckpt_path = os.path.join(checkpoint_path, 'embedding_model.ckpt')
            if os.path.exists(emb_ckpt_path):
                emb_state = torch.load(emb_ckpt_path, map_location=self.device)
                self.embedding_model.load_state_dict(emb_state)
                print(f"Loaded embedding_model from {emb_ckpt_path}")
            else:
                raise FileNotFoundError(f"embedding_model.ckpt not found in {checkpoint_path}")
            
            norm_ckpt_path = os.path.join(checkpoint_path, 'normalizer.ckpt')
            if os.path.exists(norm_ckpt_path):
                norm_state = torch.load(norm_ckpt_path, map_location=self.device)

                self.mean_var_norm.load_state_dict(norm_state, strict=False)
                print(f"Loaded normalizer from {norm_ckpt_path}")
            else:
                print(f"normalizer.ckpt not found, using uninitialized normalizer (Won't work though)")
        
        else:
            print(f"Loading single checkpoint file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'embedding_model' in checkpoint:
                self.embedding_model.load_state_dict(checkpoint['embedding_model'])
                print("Loaded embedding_model from checkpoint")
            elif 'model' in checkpoint:
                self.embedding_model.load_state_dict(checkpoint['model'])
                print("Loaded model from checkpoint")
            else:
                self.embedding_model.load_state_dict(checkpoint)
                print("Loaded weights directly from checkpoint")
            
            if 'normalizer' in checkpoint:
                self.mean_var_norm.load_state_dict(checkpoint['normalizer'], strict=False)
                print("Loaded normalizer from checkpoint")
    
    @torch.no_grad()
    def extract_embedding(self, wav_path, normalize=True):
        """
        Extract embedding from a single audio file.
        
        Args:
            wav_path: Path to wav file
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        signal, fs = torchaudio.load(wav_path)
        signal = signal.to(self.device)
        
        if signal.shape[0] > 1:
            signal = signal[0:1, :]
        
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000).to(self.device)
            signal = resampler(signal)
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if signal.dim() == 2 and signal.shape[0] == 1:
            signal = signal.unsqueeze(0)  
        
        signal = signal.squeeze(1)  
        
        feats = self.compute_features(signal)  
        
        lens = torch.ones(1).to(self.device)  
        feats = self.mean_var_norm(feats, lens)
        
        embedding = self.embedding_model(feats)  
        embedding = embedding.squeeze()  
        
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        
        embedding = embedding.cpu().numpy()
        
        return embedding
    
    def extract_embeddings_batch(self, wav_dict, normalize=True, batch_size=1):
        """
        Extract embeddings for multiple utterances.
        
        Args:
            wav_dict: Dictionary {utt_id: wav_path}
            normalize: Whether to L2-normalize embeddings
            batch_size: Process batch_size files at once (default 1 for variable length)
            
        Returns:
            embeddings_dict: Dictionary {utt_id: embedding_array}
        """
        embeddings_dict = {}
        
        print(f"Extracting embeddings for {len(wav_dict)} utterances...")
        
        for utt_id, wav_path in tqdm(wav_dict.items(), desc="Extracting"):
            try:
                embedding = self.extract_embedding(wav_path, normalize=normalize)
                embeddings_dict[utt_id] = embedding
            except Exception as e:
                print(f"Error processing {utt_id}: {e}")
                continue
        
        print(f"Successfully extracted {len(embeddings_dict)}/{len(wav_dict)} embeddings")
        
        return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description='Extract ECAPA-TDNN speaker embeddings')
    parser.add_argument('--wav_scp', type=str, required=True,
                        help='Path to Kaldi-style wav.scp file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (CKPT+XX/) or .ckpt file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to hyperparams.yaml (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip L2 normalization of embeddings')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Reading wav.scp from {args.wav_scp}")
    wav_dict = read_wav_scp(args.wav_scp)
    print(f"Found {len(wav_dict)} utterances")
    
    extractor = EmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    embeddings_dict = extractor.extract_embeddings_batch(
        wav_dict,
        normalize=not args.no_normalize
    )
    
    print(f"Saving embeddings to {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Done - Saved {len(embeddings_dict)} embeddings")
    print(f"Embedding shape: {next(iter(embeddings_dict.values())).shape}")


if __name__ == '__main__':
    main()