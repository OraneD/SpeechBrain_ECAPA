import pickle as pkl
import numpy as np
from sklearn.metrics import roc_curve
from utils import get_avg_embs



def l2norm(x, eps=1e-12):
    n = np.linalg.norm(x)
    if n < eps:
        return x * 0.0
    return x / n


def cosine_score(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    a = l2norm(a)
    b = l2norm(b)
    return float(np.dot(a, b))


def compute_eer(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    print(f"Score stats -> min: {scores.min():.4f}  max: {scores.max():.4f}  mean: {scores.mean():.4f}")

    if np.allclose(scores, scores[0]):
        print("WARNING: All scores are identical -> EER meaningless")

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100.0


def load_trials(trials_path):
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

    identical_count = 0

    for spk_id, utt_id, label in trials:
        if spk_id not in enroll_dict:
            raise KeyError(f"Enroll speaker {spk_id} not found")

        if utt_id not in trial_dict:
            raise KeyError(f"Trial utterance {utt_id} not found")

        enroll_emb = enroll_dict[spk_id]
        trial_emb = trial_dict[utt_id]

        if np.allclose(enroll_emb, trial_emb):
            identical_count += 1

        score = cosine_score(trial_emb, enroll_emb)

        if np.isnan(score):
            print("NaN score detected -> skipping")
            continue

        scores.append(score)
        labels.append(label)

    labels = np.asarray(labels)

    print(f"\nTargets: {labels.sum()}  Non-targets: {len(labels) - labels.sum()}")
    print(f"Total trials used: {len(labels)}")
    print(f"Identical enroll/trial embeddings: {identical_count}")

    eer = compute_eer(scores, labels)
    return eer



def main():
    with open("../speaker_embeddings/libri_test_enrolls_B5.pkl", "rb") as f:
        enroll_embs = pkl.load(f)
    enroll_embs = get_avg_embs(enroll_embs)

    with open("../speaker_embeddings/libri_test_trials_B5.pkl", "rb") as f:
        trial_embs = pkl.load(f)

    trials = load_trials("trials")

    eer = evaluate_eer(enroll_embs, trial_embs, trials)
    print(f"\nEER: {eer:.2f}%")


if __name__ == "__main__":
    main()
