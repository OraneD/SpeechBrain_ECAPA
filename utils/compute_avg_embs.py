
import numpy as np
from tqdm import tqdm
import pickle as pkl

def get_avg_embs(utt_embs):
    spk2_utt_dic = {}

    for utt, emb in utt_embs.items():
        spk_id = utt.split("-")[0]
        spk2_utt_dic.setdefault(spk_id, []).append(emb)

    spk2embs_avg = {}

    for spk, embs in spk2_utt_dic.items():
        avg = np.mean(embs, axis=0)

        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        spk2embs_avg[spk] = avg

    return spk2embs_avg


def main() :
    with open("../speaker_embeddings/libri_test_enrolls_B5.pkl", "rb") as f :
        test_embs = pkl.load(f)
    spk2embs = get_avg_embs(test_embs)
    print(spk2embs.keys())


if __name__ == "__main__" :
    main()

    
