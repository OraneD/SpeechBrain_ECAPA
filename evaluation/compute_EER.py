import pickle as pkl
import numpy as np
from sklearn.metrics import roc_curve
from utils import get_avg_embs


def cosine_score(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100.0


def load_trials(trials_path):
    """
    Returns list of (spk_id, utt_id, label)
    label: 1 = target, 0 = nontarget
    """
    trials = []
    with open(trials_path) as f:
        for line in f:
            spk, utt, lab = line.strip().split()
            label = 1 if lab == "target" else 0
            trials.append((spk, utt, label))
    return trials


def evaluate_eer(enroll_dict, trial_dict, trials):
    scores = []
    labels = []

    for spk_id, utt_id, label in trials:
        if spk_id not in enroll_dict:
            raise KeyError(f"Enroll speaker {spk_id} not found")

        if utt_id not in trial_dict:
            raise KeyError(f"Trial utterance {utt_id} not found")

        enroll_emb = enroll_dict[spk_id]
        trial_emb = trial_dict[utt_id]

        score = cosine_score(trial_emb, enroll_emb)
        scores.append(score)
        labels.append(label)

    print(f"{sum(labels)} Targets - {len(labels) - sum(labels)} Non-Targets")
    print(f"{len(labels)} Trials total")

    eer = compute_eer(scores, labels)
    return eer


def main():
    with open("../speaker_embeddings/libri_test_enrolls_B5_ECAPA_speechbrain.pkl", "rb") as f:
        enroll_embs = pkl.load(f)
    enroll_embs = get_avg_embs(enroll_embs)

    with open("../speaker_embeddings/libri_test_trials_B5_ECAPA_speechbrain.pkl", "rb") as f:
        trial_embs = pkl.load(f)

    trials = load_trials("trials") 

    eer = evaluate_eer(enroll_embs, trial_embs, trials)
    print(f"EER: {eer:.2f}%")


if __name__ == "__main__":
    main()
