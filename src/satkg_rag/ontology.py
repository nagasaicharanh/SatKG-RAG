from __future__ import annotations

from pathlib import Path
import re

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef, XSD

from .models import Triple

SAT = Namespace("http://example.org/satkg#")


class OntologyManager:
    def __init__(self) -> None:
        self.graph = Graph()
        self.graph.bind("sat", SAT)
        self._build_schema()

    def _build_schema(self) -> None:
        classes = [
            "Satellite",
            "Component",
            "Anomaly",
            "Sensor",
            "MissionPhase",
            "TelemetryParameter",
        ]
        for cls in classes:
            self.graph.add((SAT[cls], RDF.type, OWL.Class))

        object_properties = {
            "hasComponent": ("Satellite", "Component"),
            "monitors": ("Sensor", "TelemetryParameter"),
            "triggeredBy": ("Anomaly", "Sensor"),
            "causedBy": ("Anomaly", "Component"),
            "operatesDuring": ("Satellite", "MissionPhase"),
            "mentionedWith": (None, None),
            "relatedTo": (None, None),
        }
        for prop, (domain, range_) in object_properties.items():
            self.graph.add((SAT[prop], RDF.type, OWL.ObjectProperty))
            if domain:
                self.graph.add((SAT[prop], RDFS.domain, SAT[domain]))
            if range_:
                self.graph.add((SAT[prop], RDFS.range, SAT[range_]))

        data_properties = {
            "hasTemperatureThreshold": ("Component", XSD.string),
            "hasNominalRange": ("TelemetryParameter", XSD.string),
            "hasTimestamp": ("Anomaly", XSD.dateTime),
        }
        for prop, (domain, range_) in data_properties.items():
            self.graph.add((SAT[prop], RDF.type, OWL.DatatypeProperty))
            self.graph.add((SAT[prop], RDFS.domain, SAT[domain]))
            self.graph.add((SAT[prop], RDFS.range, range_))

    def _sanitize(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_+-]+", "_", value.strip())
        return cleaned.strip("_") or "Unknown"

    def _class_for_value(self, value: str, predicate: str, is_subject: bool) -> URIRef | None:
        predicate_map = {
            "hasComponent": ("Satellite", "Component"),
            "monitors": ("Sensor", "TelemetryParameter"),
            "triggeredBy": ("Anomaly", "Sensor"),
            "causedBy": ("Anomaly", "Component"),
            "operatesDuring": ("Satellite", "MissionPhase"),
            "hasTemperatureThreshold": ("Component", None),
            "hasNominalRange": ("TelemetryParameter", None),
            "hasTimestamp": ("Anomaly", None),
        }
        mapped = predicate_map.get(predicate)
        if mapped:
            class_name = mapped[0] if is_subject else mapped[1]
            return SAT[class_name] if class_name else None
        lower_value = value.casefold()
        if "anomaly" in lower_value:
            return SAT.Anomaly
        if "sensor" in lower_value:
            return SAT.Sensor
        if "temperature" in lower_value or "telemetry" in lower_value:
            return SAT.TelemetryParameter
        if re.match(r"^[A-Z]{2,}-\d+", value):
            return SAT.Satellite
        return None

    def add_triple(self, triple: Triple) -> None:
        subj = URIRef(SAT[self._sanitize(triple.subject)])
        pred = URIRef(SAT[self._sanitize(triple.predicate)])
        obj_text = triple.object.strip()
        subj_class = self._class_for_value(triple.subject, triple.predicate, is_subject=True)
        obj_class = self._class_for_value(obj_text, triple.predicate, is_subject=False)
        if subj_class:
            self.graph.add((subj, RDF.type, subj_class))
        if obj_class:
            obj = URIRef(SAT[self._sanitize(obj_text)])
            self.graph.add((obj, RDF.type, obj_class))
        elif obj_text and obj_text[0].isupper():
            obj = URIRef(SAT[self._sanitize(obj_text)])
        else:
            obj = Literal(obj_text)
        self.graph.add((subj, pred, obj))

    def add_triples(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.add_triple(triple)

    def serialize_turtle(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(output_path), format="turtle")

    def turtle(self) -> str:
        return self.graph.serialize(format="turtle")

    def sparql(self, query: str):
        return self.graph.query(query)


def build_owlready_world(ontology_iri: str = "http://example.org/satkg.owl"):
    from owlready2 import DatatypeProperty, ObjectProperty, Thing, get_ontology

    onto = get_ontology(ontology_iri)
    with onto:
        class Satellite(Thing):
            pass

        class Component(Thing):
            pass

        class Anomaly(Thing):
            pass

        class Sensor(Thing):
            pass

        class MissionPhase(Thing):
            pass

        class TelemetryParameter(Thing):
            pass

        class hasComponent(ObjectProperty):
            domain = [Satellite]
            range = [Component]

        class monitors(ObjectProperty):
            domain = [Sensor]
            range = [TelemetryParameter]

        class triggeredBy(ObjectProperty):
            domain = [Anomaly]
            range = [Sensor]

        class causedBy(ObjectProperty):
            domain = [Anomaly]
            range = [Component]

        class operatesDuring(ObjectProperty):
            domain = [Satellite]
            range = [MissionPhase]

        class hasTemperatureThreshold(DatatypeProperty):
            domain = [Component]
            range = [float]

        class hasNominalRange(DatatypeProperty):
            domain = [TelemetryParameter]
            range = [str]

        class hasTimestamp(DatatypeProperty):
            domain = [Anomaly]
            range = [str]
    return onto
