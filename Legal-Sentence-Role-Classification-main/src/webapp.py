"""
Creates a Uvicorn webapp backend
@author: Philipp
"""

import csv
import os
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import src.classification.prediction as classifier
import src.finding_aid as finding_aid
import src.pii_filter as pii_filter
import src.sentence_encoder as sentence_encoder
from src import segmentation_pipeline as segmentation_pipeline
from src.classification.nn_models import LSTM_Net


class Document(BaseModel):
    text: str
    pseudonymize: bool = False


class Correction(BaseModel):
    sentence: str
    predicted_role: str
    corrected_role: str


class FindingAidRequest(BaseModel):
    sentences: List[dict]
    title: str = ""
    decision_date: str = ""


class AttributionRequest(BaseModel):
    sentence: str


app = FastAPI()

templates = Jinja2Templates(directory="web_app_templates")


@app.post("/doc")
async def perform_eval(document: Document):
    # document contains the text submitted via the webpage
    # 1. optionally pseudonymize PII before processing
    text = pii_filter.pseudonymize(document.text) if document.pseudonymize else document.text
    # 2. segment into sentences
    sentences = segmentation_pipeline.run_segmenter(text)
    # 3. encode with bert
    encoded_sentences = sentence_encoder.sentence_bert_embeddings(sentences, model=1)
    # 3 get model and do prediction
    path = os.path.dirname(os.path.abspath(__file__))
    weight_path = path + "\..\data\model_weights\LSTM Net_balanced_DICE_batch_size_1.dat"
    predicted_sentences = classifier.predict_role(encoded_sentences, LSTM_Net(), weight_path)
    # 4. sort sentences by occuring in document
    predicted_sentences = predicted_sentences.sort_values('start_char', ascending=True)

    response = []
    predicted_sentences['prob'] = predicted_sentences['prob'].round(2)
    predicted_sentences['entropy'] = predicted_sentences['entropy'].round(3)

    for index, sentence in predicted_sentences.iterrows():
        uncertain = bool(sentence['uncertain'])
        response.append({
            "sentence": sentence['text'],
            "role": "UNCERTAIN" if uncertain else sentence['role'],
            "suggested_role": sentence['role'],
            "prob": sentence['prob'],
            "entropy": sentence['entropy'],
            "uncertain": uncertain,
        })

    return response


@app.post("/correct")
async def log_correction(correction: Correction):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'correction_log.csv'
    )
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['timestamp', 'sentence', 'predicted_role', 'corrected_role']
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.utcnow().isoformat(),
            'sentence': correction.sentence,
            'predicted_role': correction.predicted_role,
            'corrected_role': correction.corrected_role,
        })
    return {"status": "logged"}


@app.post("/finding-aid")
async def generate_finding_aid(req: FindingAidRequest):
    xml = finding_aid.generate_ead(
        classified_sentences=req.sentences,
        title=req.title,
        decision_date=req.decision_date,
    )
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="finding_aid.ead.xml"'},
    )


@app.get("/contestation")
async def contestation_report():
    """Return label-contestation statistics from the correction audit log."""
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'correction_log.csv'
    )
    if not os.path.isfile(log_path):
        return {"error": "No corrections logged yet."}

    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"total_corrections": 0}

    total = len(rows)

    # Per-role disagreement counts
    role_contested: dict[str, int] = {}
    role_corrected_to: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}

    for r in rows:
        pred = r['predicted_role']
        corr = r['corrected_role']
        role_contested[pred] = role_contested.get(pred, 0) + 1
        role_corrected_to[corr] = role_corrected_to.get(corr, 0) + 1
        if pred not in confusion:
            confusion[pred] = {}
        confusion[pred][corr] = confusion[pred].get(corr, 0) + 1

    # Most contested role (most often overridden by archivist)
    most_contested = max(role_contested, key=role_contested.get)

    return {
        "total_corrections": total,
        "corrections_by_predicted_role": role_contested,
        "corrections_to_role": role_corrected_to,
        "most_contested_role": most_contested,
        "confusion_pairs": confusion,
    }


@app.post("/attribution")
async def word_attribution(req: AttributionRequest):
    """Compute word-level saliency for a single sentence using input × gradient."""
    path = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(path, '..', 'data', 'model_weights',
                               'LSTM Net_balanced_DICE_batch_size_1.dat')
    words = classifier.compute_word_attribution(
        req.sentence, LSTM_Net(), weight_path
    )
    return {"words": words}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



