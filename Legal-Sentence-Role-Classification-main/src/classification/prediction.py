"""
Contains methods for straight forward prediction using the trained models,
aswell as creating graphs like confusion matrices

@author: Philipp
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.classification.custom_pytorch_dataset import CustomDataset
from src.classification.nn_models import LSTM_Net

# ── Cached models for word-level attribution ──
_bert_model = None
_bert_tokenizer = None


def _get_bert():
    """Lazy-load and cache the LegalBERT model + tokenizer for attribution."""
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from transformers import AutoTokenizer, AutoModel
        _bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        _bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        _bert_model.eval()
    return _bert_model, _bert_tokenizer


def compute_word_attribution(sentence_text, lstm_model, weight_path):
    """Compute word-level saliency scores using input × gradient through BERT→LSTM.

    Returns a list of {"word": str, "score": float} where score is in [0, 1].
    """
    bert, tokenizer = _get_bert()

    device = torch.device("cpu")
    lstm_model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    lstm_model.to(device)
    lstm_model.eval()

    # Tokenize
    encoded = tokenizer(
        sentence_text, return_tensors="pt", truncation=True,
        max_length=512, padding=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    # Get input embeddings with gradient tracking
    input_embeds = bert.embeddings.word_embeddings(encoded["input_ids"])
    input_embeds = input_embeds.detach().requires_grad_(True)

    # Forward through BERT using custom embeddings
    outputs = bert(
        inputs_embeds=input_embeds,
        attention_mask=encoded["attention_mask"],
    )

    # Mean pooling (matches sentence-transformers default)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    sentence_emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

    # Forward through LSTM
    lstm_input = sentence_emb.unsqueeze(0)  # [1, 1, 768]
    logits = lstm_model(lstm_input)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()

    # Backward from predicted class
    probs[0, pred_idx].backward()

    # Input × gradient per token
    grad = input_embeds.grad[0]  # [seq_len, 768]
    importance = (grad * input_embeds.detach()[0]).sum(dim=-1).abs()

    # Zero out special tokens
    importance[0] = 0   # [CLS]
    importance[-1] = 0  # [SEP]

    # Normalize to [0, 1]
    max_imp = importance.max()
    if max_imp > 0:
        importance = importance / max_imp

    # Merge subword tokens back to words
    words = []
    scores = []
    current_word = ""
    current_scores = []

    for token, score in zip(tokens, importance.tolist()):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_scores.append(score)
        else:
            if current_word:
                words.append(current_word)
                scores.append(max(current_scores))
            current_word = token
            current_scores = [score]
    if current_word:
        words.append(current_word)
        scores.append(max(current_scores))

    return [{"word": w, "score": round(s, 3)} for w, s in zip(words, scores)]


### The dataframe needs an column "embedding" and "label_encoded" e.g. for confusion matrix
###returns previous dataframe incl. now predicted labels and probability
def predict_role_with_true_label(dataframe, model, weight_path):
    dataframe = dataframe.reset_index(drop=True)
    data = CustomDataset(dataframe)
    data_loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=1)

    with torch.no_grad():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        model.eval()
        pred_rows = []
        for sample_batched in data_loader:
            sentences = sample_batched['sentence'].float().to(device)
            labels = sample_batched['label'].float()
            indices = sample_batched['index']
            pred = model(sentences)
            probs = (F.softmax(pred, dim=1)).to('cpu')
            predictions = (torch.argmax(probs, dim=1).numpy())
            for i, idx in enumerate(indices):
                pred_rows.append(
                    {"Index": idx.item(), "True Label": labels.numpy()[i].argmax(), "Predicted Label": predictions[i],
                     "Probability": probs[i].numpy()[predictions[i]]})

        df_predictions = pd.DataFrame(pred_rows)
        df_predictions = pd.merge(left=dataframe, right=df_predictions,
                                  left_index=True,
                                  right_on="Index", how="outer")
        # maps the label numbers to text
        mapping = pd.DataFrame([
            {'Predicted Role': 'Citation', 'label': 0},
            {'Predicted Role': 'Evidence', 'label': 1},
            {'Predicted Role': 'Finding', 'label': 2},
            {'Predicted Role': 'Legal Rule', 'label': 3},
            {'Predicted Role': 'Reasoning', 'label': 4},
            {'Predicted Role': 'Sentence', 'label': 5},
        ])
        print(df_predictions.info())
        print(mapping.info())
        # merges mapping with predictions
        df_predictions = pd.merge(left=df_predictions, right=mapping, left_on='Predicted Label',right_on="label", how="outer")
        print(df_predictions.info())

        return df_predictions

### needs only a dataframe with 'embedding'
### returns old dataframe now including 'label' the predicted labels and 'prob' = pseudo probabilities
def predict_role(dataframe, model, weight_path):
    dataframe = dataframe.reset_index(drop=True)

    data = CustomDataset(dataframe)
    data_loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=1)

    with torch.no_grad():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        pred_rows = []
        for sample_batched in data_loader:
            sentences = sample_batched['sentence'].float().to(device)
            indices = sample_batched['index']
            pred = model(sentences)
            probs = (F.softmax(pred, dim=1)).to('cpu')
            predictions = (torch.argmax(probs, dim=1).numpy())

            #merges old dataframe with predictions
            for i, idx in enumerate(indices):
                prob_dist = probs[i].numpy()
                max_prob = float(prob_dist[predictions[i]])
                entropy = float(-np.sum(prob_dist * np.log(prob_dist + 1e-10)))
                pred_rows.append(
                    {"index": idx.item(), "label": predictions[i], "prob": max_prob,
                     "entropy": entropy, "uncertain": max_prob < 0.65})
        df_predictions = pd.DataFrame(pred_rows)
        df_predictions = pd.merge(left=dataframe, right=df_predictions,
                                  left_index=True,
                                  right_on="index", how="inner")
        #maps the label numbers to text
        mapping = pd.DataFrame([
            {'role': 'Citation', 'label': 0},
            {'role': 'Evidence', 'label': 1},
            {'role': 'Finding', 'label': 2},
            {'role': 'Legal Rule', 'label': 3},
            {'role': 'Reasoning', 'label': 4},
            {'role': 'Sentence', 'label': 5},
        ])

        #merges mapping with predictions
        df_predictions = pd.merge(left=df_predictions, right=mapping, on="label", how="inner")
        print(df_predictions.head())
        return df_predictions

#print confusion matrix and classification report on given dataset and model
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # weight_path= "../../data/model_weights/balanced_Logistic Regression_DICE_F1_Batch1.dat"
    # weight_path= "../../data/model_weights/balanced_MLP_DICE_F1.dat"
    weight_path = "../../data/model_weights/LSTM Net_balanced_DICE_batch_size_1.dat"
    # weight_path= "../../data/model_weights/Logistic Regression_balanced_DICE_batch_size_1.dat"

    df_sentences = pickle.load(open('../../data/sentences_balanced_legalBERT.p', 'rb'))




    predictions = predict_role_with_true_label(df_sentences[df_sentences.Split == "Test"], LSTM_Net(), weight_path)
    print(predictions.info())
    # predictions.to_csv("../../data/graphs/error_analysis.csv")
    predictions['label_x'] = predictions['label_x'].map(lambda x: x.replace('Sentence','').replace('LegalRule','Legal Rule') if x!='Sentence'else x)
    # predictions = predictions.drop(['label_y'])
    predictions=predictions.rename(columns={'label_x':'Ground Truth Label'})
    print(classification_report(predictions['Ground Truth Label'], predictions['Predicted Role']))
    contingency_matrix = pd.crosstab(predictions['Ground Truth Label'], predictions['Predicted Role'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = sn.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../../data/graphs/confusion_matrix_lstm_test.png")
    plt.show()
