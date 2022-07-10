import json
from sensebert import SenseBert
import tensorflow as tf
from wic_util.data_loader import WicFewShotLoader, InputExample

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data_type = "test"

def extract_rep_single(sent_id, pos, mat):
    word_st, word_ed = pos
    rep = mat[sent_id][word_st:word_ed]
    # print(rep.shape)
    sum_rep = rep.sum(axis=0).reshape((1, -1))
    # print(sum_rep.shape)
    return sum_rep


def extract_rep(mat, tgt_positions):
    word_a = extract_rep_single(0, tgt_positions[0], mat)
    word_b = extract_rep_single(1, tgt_positions[1], mat)
    score = cosine_similarity(word_a, word_b)
    return word_a, word_b, score


def save_results(mat, tag):
    np_mat = np.asarray(mat)
    # print(np_mat.shape)
    path = "output_dir/"+data_type+"_"+tag
    print(f"saving to {path}.npy")
    np.save(path, mat)


if __name__ == '__main__':
    with open("config.json") as fp:
        cfg = json.load(fp)
        model_path = cfg["model"]
        data_path = cfg["data"]
        wic_data = WicFewShotLoader(data_path)

    word_embeds = []
    word_logits = []
    embed_scores = []
    logit_scores = []
    labels = []

    with tf.Session() as session:
        sensebert_model = SenseBert(model_path, session=session)  # or sensebert-large-uncased
        for sample in wic_data.test:
            input_ids, input_mask, tgt_positions = sensebert_model.tokenize([sample.text_a, sample.text_b], \
                [sample.word_a_char_pos, sample.word_b_char_pos])
            model_outputs = sensebert_model.run(input_ids, input_mask)
            contextualized_embeddings, mlm_logits, supersense_logits = model_outputs  # these are NumPy arrays
            word_a_embed, word_b_embed, embed_score = extract_rep(contextualized_embeddings, tgt_positions)
            word_a_logit, word_b_logit, logit_score = extract_rep(mlm_logits, tgt_positions)
            word_embeds.append(np.vstack([word_a_embed, word_b_embed]))
            word_logits.append(np.vstack([word_a_logit, word_b_logit]))
            embed_scores.append(embed_score)
            logit_scores.append(logit_score)
            labels.append(sample.label)
    
    save_results(word_embeds, "word_embeds")
    save_results(word_logits, "word_logits")
    save_results(np.vstack(embed_scores), "embed_scores")
    save_results(np.vstack(logit_scores), "logit_scores")
    save_results(labels, "labels")
    