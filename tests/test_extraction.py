from extraction.mock import extract_heuristically
from extraction.extractor import ExtractionService


def test_mock_extraction_multihop():
    data = extract_heuristically("Alice founded Company X. Company X acquired Company Y. Company Y developed Product Z.")
    rels = {(r["source"], r["relation"], r["target"]) for r in data["relationships"]}
    assert ("Alice", "founded", "Company X") in rels
    assert ("Company X", "acquired", "Company Y") in rels
    assert ("Company Y", "developed", "Product Z") in rels


def test_invalid_json_raises():
    svc = ExtractionService()
    try:
        svc._parse_json("not json")
    except ValueError as exc:
        assert "no JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON failure")


def test_founded_by_extraction():
    data = extract_heuristically("Acme Robotics was founded by Arjun Mehta in 2018.")
    rels = {(r["source"], r["relation"], r["target"]) for r in data["relationships"]}
    assert ("Acme Robotics", "founded_by", "Arjun Mehta") in rels
