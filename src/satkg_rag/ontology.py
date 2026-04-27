from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

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
            self.graph.add((SAT[cls], RDF.type, RDFS.Class))

        object_properties = [
            "hasComponent",
            "monitors",
            "triggeredBy",
            "causedBy",
            "operatesDuring",
        ]
        for prop in object_properties:
            self.graph.add((SAT[prop], RDF.type, RDF.Property))

        data_properties = [
            "hasTemperatureThreshold",
            "hasNominalRange",
            "hasTimestamp",
        ]
        for prop in data_properties:
            self.graph.add((SAT[prop], RDF.type, RDF.Property))

    def _sanitize(self, value: str) -> str:
        return value.strip().replace(" ", "_").replace("/", "_")

    def add_triple(self, triple: Triple) -> None:
        subj = URIRef(SAT[self._sanitize(triple.subject)])
        pred = URIRef(SAT[self._sanitize(triple.predicate)])
        obj_text = triple.object.strip()
        if obj_text and obj_text[0].isupper():
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
