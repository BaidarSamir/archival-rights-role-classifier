"""
PII detection and pseudonymization for BVA decision text.
Uses spaCy NER to identify and optionally replace sensitive entities
before encoding or storing, in line with data-protection principles.
@author: added for archival rights role classifier
"""
import spacy

_nlp = None


def _get_nlp():
    """Lazily load the spaCy model (NER only) so import is cheap."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(
            "en_core_web_lg",
            disable=["tagger", "parser", "senter", "attribute_ruler", "lemmatizer"],
        )
    return _nlp


SENSITIVE_LABELS = {"PERSON", "DATE", "GPE", "ORG", "CARDINAL"}

ROLE_MAP = {
    "PERSON": "[VETERAN/PHYSICIAN]",
    "DATE": "[DATE]",
    "GPE": "[LOCATION]",
    "ORG": "[ORGANIZATION]",
    "CARDINAL": "[NUMBER]",
}


def detect_pii(text):
    """Return a list of detected PII entities with span info.

    Each entry is a dict with keys: text, label, start, end.
    """
    doc = _get_nlp()(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in SENSITIVE_LABELS
    ]


def pseudonymize(text):
    """Replace sensitive named entities in text with neutral placeholders.

    Entities are replaced right-to-left to preserve character offsets.
    """
    doc = _get_nlp()(text)
    result = text
    for ent in reversed(doc.ents):
        if ent.label_ in ROLE_MAP:
            result = result[: ent.start_char] + ROLE_MAP[ent.label_] + result[ent.end_char :]
    return result
