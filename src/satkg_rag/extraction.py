from __future__ import annotations

import json
import logging
import re
import re

from .models import EntityMention, Triple, TripleBatch

logger = logging.getLogger(__name__)


def load_ner_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required for entity extraction. Install project dependencies "
            "and download the configured model, for example: "
            "pip install -e . && python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning("spaCy model %s is not installed; using a lightweight satellite-domain EntityRuler.", model_name)
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(
            [
                {"label": "COMPONENT", "pattern": "battery"},
                {"label": "COMPONENT", "pattern": "battery subsystem"},
                {"label": "COMPONENT", "pattern": "thermal control system"},
                {"label": "SENSOR", "pattern": "ThermalSensor"},
                {"label": "SENSOR", "pattern": "thermal sensor"},
                {"label": "ANOMALY", "pattern": "thermal anomaly"},
                {"label": "ANOMALY", "pattern": "safe mode"},
                {"label": "TELEMETRY_PARAMETER", "pattern": "BatteryTemperature"},
                {"label": "TELEMETRY_PARAMETER", "pattern": "battery temperature"},
                {"label": "MISSION_PHASE", "pattern": "LEO"},
                {"label": "MISSION_PHASE", "pattern": "low Earth orbit"},
                {"label": "ORG", "pattern": "Airbus"},
                {"label": "ORG", "pattern": "DLR"},
            ]
        )
        return nlp


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


def extract_triples_with_rules(text: str, entities: list[EntityMention]) -> TripleBatch:
    triples: list[Triple] = []
    normalized = text.casefold()
    entity_by_label: dict[str, list[str]] = {}
    for entity in entities:
        entity_by_label.setdefault(entity.label, []).append(entity.text)

    for satellite_id in sorted(set(re.findall(r"\b[A-Z]{2,}-\d+[A-Z0-9_-]*\b", text))):
        entity_by_label.setdefault("SATELLITE", []).append(satellite_id)
    domain_terms = {
        "COMPONENT": ["battery", "battery subsystem", "payload", "solar array", "reaction wheel", "thermal control system"],
        "SENSOR": ["thermal sensor", "gyroscope", "star tracker"],
        "ANOMALY": ["thermal anomaly", "safe mode", "thermal drift", "power anomaly"],
        "TELEMETRY_PARAMETER": ["battery temperature", "BatteryTemperature", "voltage", "current"],
        "MISSION_PHASE": ["LEO", "low Earth orbit", "commissioning", "eclipse"],
    }
    for label, terms in domain_terms.items():
        for term in terms:
            if term.casefold() in normalized:
                entity_by_label.setdefault(label, []).append(term)

    sensors = entity_by_label.get("SENSOR", [])
    parameters = entity_by_label.get("TELEMETRY_PARAMETER", [])
    components = entity_by_label.get("COMPONENT", [])
    anomalies = entity_by_label.get("ANOMALY", [])
    phases = entity_by_label.get("MISSION_PHASE", [])

    for sensor in sensors:
        for parameter in parameters:
            triples.append(Triple(subject=sensor, predicate="monitors", object=parameter, confidence=0.7))

    for anomaly in anomalies:
        for component in components:
            triples.append(Triple(subject=anomaly, predicate="causedBy", object=component, confidence=0.55))
        for sensor in sensors:
            triples.append(Triple(subject=anomaly, predicate="triggeredBy", object=sensor, confidence=0.55))

    satellites = sorted(set(entity_by_label.get("SATELLITE", [])))
    for satellite in satellites:
        for component in components:
            triples.append(Triple(subject=satellite, predicate="hasComponent", object=component, confidence=0.6))
        for phase in phases:
            triples.append(Triple(subject=satellite, predicate="operatesDuring", object=phase, confidence=0.6))

    if "temperature" in normalized and "threshold" in normalized:
        for component in components:
            triples.append(
                Triple(subject=component, predicate="hasTemperatureThreshold", object="mentioned", confidence=0.45)
            )

    entity_names = []
    seen_entities = set()
    for values in entity_by_label.values():
        for value in values:
            key = value.casefold()
            if key not in seen_entities:
                seen_entities.add(key)
                entity_names.append(value)
    if not triples and len(entity_names) >= 2:
        for subject, object_value in zip(entity_names, entity_names[1:]):
            triples.append(Triple(subject=subject, predicate="mentionedWith", object=object_value, confidence=0.35))

    unique: dict[tuple[str, str, str], Triple] = {}
    for triple in triples:
        unique[(triple.subject, triple.predicate, triple.object)] = triple
    return TripleBatch(triples=list(unique.values()))


def _triple_prompt(text: str) -> str:
    return (
        "Extract factual subject-predicate-object triples from the text. "
        "Return ONLY JSON in this schema: "
        '{"triples":[{"subject":"...","predicate":"...","object":"...","confidence":0.0,"source":"..."}]}.\n'
        f"Text:\n{text}"
    )


def _normalize_triple(raw_triple: object) -> dict[str, object] | None:
    if not isinstance(raw_triple, dict):
        return None

    subject = str(raw_triple.get("subject", "")).strip()
    predicate = str(raw_triple.get("predicate", raw_triple.get("relation", ""))).strip()
    object_value = str(raw_triple.get("object", raw_triple.get("obj", ""))).strip()

    if not subject or not predicate or not object_value:
        return None

    confidence = raw_triple.get("confidence")
    parsed_confidence = None
    if confidence is not None:
        try:
            parsed_confidence = float(confidence)
        except (TypeError, ValueError):
            parsed_confidence = None

    source = raw_triple.get("source")
    parsed_source = None if source is None else str(source)

    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "confidence": parsed_confidence,
        "source": parsed_source,
    }


def extract_triples_with_ollama(text: str, model_name: str = "mistral") -> TripleBatch:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The ollama Python package is required for LLM triple extraction. "
            "Install project dependencies and ensure the local Ollama service is running."
        ) from exc

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": _triple_prompt(text)}],
        format="json",
    )
    raw = response["message"]["content"]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response must be a JSON object")

    raw_triples = parsed.get("triples", [])
    if not isinstance(raw_triples, list):
        raise ValueError("Ollama response must include a list field named 'triples'")

    normalized_triples: list[dict[str, object]] = []
    skipped = 0
    for raw_triple in raw_triples:
        normalized = _normalize_triple(raw_triple)
        if normalized is None:
            skipped += 1
            continue
        normalized_triples.append(normalized)

    if skipped:
        logger.warning("Skipped %d malformed triples from Ollama output.", skipped)

    return TripleBatch.model_validate({"triples": normalized_triples})
