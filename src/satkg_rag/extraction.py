from __future__ import annotations

import json

import ollama
import spacy

from .models import EntityMention, TripleBatch


def load_ner_model(model_name: str):
    return spacy.load(model_name)


def extract_entities(text: str, nlp) -> list[EntityMention]:
    doc = nlp(text)
    return [
        EntityMention(
            text=ent.text,
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
        )
        for ent in doc.ents
    ]


def _triple_prompt(text: str) -> str:
    return (
        "Extract factual subject-predicate-object triples from the text. "
        "Return ONLY JSON in this schema: "
        '{"triples":[{"subject":"...","predicate":"...","object":"...","confidence":0.0,"source":"..."}]}.\n'
        f"Text:\n{text}"
    )


def extract_triples_with_ollama(text: str, model_name: str = "mistral") -> TripleBatch:
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": _triple_prompt(text)}],
        format="json",
    )
    raw = response["message"]["content"]
    parsed = json.loads(raw)
    return TripleBatch.model_validate(parsed)
