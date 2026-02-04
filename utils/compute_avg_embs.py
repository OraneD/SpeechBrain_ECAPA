
import numpy as np
from tqdm import tqdm
import pickle as pkl

def get_avg_embs(utt_embs):
    spk2_utt_dic = {}
    for utt, emb in tqdm(utt_embs.items()) : 
        spk_id = utt.split("-")[0]
        if spk_id not in spk2_utt_dic : 
            spk2_utt_dic[spk_id] = []
        spk2_utt_dic[spk_id].append(emb)
    
    spk2embs_avg = {}
    for spk, embs in tqdm(spk2_utt_dic.items()) :
        spk2embs_avg[spk] = np.mean(embs, axis=0)
    return spk2embs_avg

def main() :
    with open("../speaker_embeddings/libri_test_enrolls_B5_ECAPA_speechbrain.pkl", "rb") as f :
        test_embs = pkl.load(f)
    print(get_avg_embs(test_embs))


if __name__ == "__main__" :
    main()

    
