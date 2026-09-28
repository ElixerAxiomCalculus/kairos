from kairos.controller.turns import TurnClassifier, parse_transform
from kairos.schemas.models import AnswerBlock, AnswerVersion, Claim
from kairos.streaming.frame import apply_repairs, segment
from kairos.synthesis.transform import TransformOutcome, verify_transform
from kairos.textutil import anchors, norm, rbo


def test_rbo_bounds():
    assert rbo(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert rbo(["a", "b"], ["x", "y"]) == 0.0
    assert 0 < rbo(["a", "b", "c"], ["a", "c", "b"]) < 1


def test_norm_canonicalises_headcount_nouns():
    assert norm("participants") == norm("people") == "people"
    assert norm("bookings") == norm("booking")


def test_contrast_facets_are_corpus_derived(ctx):
    groups = ctx.corpus.contrast_groups
    assert any({"international", "standard"} <= members for members in groups.values())
    intl = [c for c in ctx.corpus.chunks if "international" in c.facets.values()]
    assert {c.doc_id for c in intl} >= {"D03", "D04", "D05", "D06"}
    # "Standard equipment" is not an event/travel heading, so it must not carry the facet
    assert not ctx.corpus.chunk("D07:C01").facets


def test_segmentation_splits_requests_and_keeps_constraints(ctx):
    spans = segment("I need to book a workshop venue in Pune for 30 people, tell me the cancellation policy "
                    "and whether catering is available.", ctx.corpus)
    needs = [s.text for s in spans if s.kind == "NEED"]
    assert len(needs) == 3
    assert "for 30 people" in needs[0]


def test_self_repair_replaces_same_class_slot(ctx):
    text, repairs = apply_repairs("Which rooms can seat 100 people in Pune, sorry, Mumbai?", ctx.corpus)
    assert "Mumbai" in text and "Pune" not in text
    assert repairs == [{"kind": "place", "from": "Pune", "to": "Mumbai"}]


def test_turn_classifier_separates_presentation_from_information(ctx):
    tc = TurnClassifier(ctx.corpus)
    vocab = {"cancellation", "policy", "venue", "catering"}
    assert tc.classify("Make that shorter.", True, vocab, True).turn_type == "TRANSFORM"
    assert tc.classify("Translate that into Hindi.", True, vocab, True).turn_type == "TRANSFORM"
    assert tc.classify("Actually, this is for an international event.", True, vocab, True).turn_type == "CONSTRAINT"
    assert tc.classify("What is the per diem for domestic travel?", True, vocab, True).turn_type == "NEW_INFO"
    assert tc.classify("Summarize the reimbursement policy.", False, set(), True).turn_type == "NEW_INFO"


def test_parse_transform():
    assert parse_transform("Give it to me in two bullets")["bullets"] == 2
    assert "condense" in parse_transform("make that shorter")["ops"]
    spec = parse_transform("translate that into Hindi")
    assert spec["language"] == "Hindi" and "translate" in spec["ops"]


def test_anchors_extract_numbers_and_names():
    a = anchors("The standard lunch buffet costs ₹850 per person at Banyan Hall.")
    assert "₹850" in a and "hall" in a


def _version(claims):
    return AnswerVersion(1, None, "NEW_INFO", "SEMANTIC", "u1", 1,
                         [AnswerBlock("I01", "t", "paragraph", [c.claim_id for c in claims])],
                         {c.claim_id: c for c in claims})


def test_transform_firewall_rejects_new_numbers_and_citations():
    src = Claim("C01", "L01", "Cancellations 14 days before are free.", "I01", ["D03:C03"], "q", 1.0)
    parent = _version([src])
    bad = Claim("C02", "L01", "Cancellations 10 days before are free.", "I01", ["D03:C03"], "q", 1.0,
                derived_from=["C01"], status="TRANSFORMED")
    foreign = Claim("C03", "L01", "Cancellations 14 days before are free.", "I01", ["D09:C01"], "q", 1.0,
                    derived_from=["C01"], status="TRANSFORMED")
    check = verify_transform(parent, TransformOutcome(True, [], {"C02": bad, "C03": foreign}))
    rules = {(v["claim_id"], v["rule"]) for v in check["violations"]}
    assert ("C02", "NEW_NUMERIC_ANCHOR") in rules
    assert ("C03", "NEW_CITATION") in rules
    assert not check["passed"]
