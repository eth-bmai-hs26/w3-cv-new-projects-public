import pytest

from backend import rules
from backend.rules import APPROVE, REJECT, REVIEW, RuleConfig, decide


def test_default_rule_is_85_percent_with_5_point_band():
    cfg = RuleConfig()
    assert cfg.threshold == 0.85 and cfg.review_band == 0.05


@pytest.mark.parametrize("uf, expected", [
    (1.00, APPROVE), (0.95, APPROVE), (0.90, APPROVE),      # >= threshold + band
    (0.899, REVIEW), (0.85, REVIEW), (0.80, REVIEW),        # inside the band
    (0.799, REJECT), (0.40, REJECT), (0.0, REJECT),
])
def test_threshold_and_band(uf, expected):
    assert decide(uf, mask_certainty=1.0).verdict == expected


def test_without_band_threshold_is_inclusive():
    cfg = RuleConfig(review_band=0.0, min_confidence=0.0)
    assert decide(0.85, 1.0, cfg).verdict == APPROVE
    assert decide(0.8499, 1.0, cfg).verdict == REJECT


def test_low_mask_certainty_sends_clear_tile_to_review():
    v = decide(0.97, mask_certainty=0.05)
    assert v.verdict == REVIEW and "confidence" in v.reason.lower()
    assert decide(0.97, mask_certainty=0.95).verdict == APPROVE


def test_confidence_grows_with_distance_from_threshold():
    near = decide(0.905, 1.0).confidence
    far = decide(0.99, 1.0).confidence
    assert 0 < near < far <= 1


def test_no_tile_is_flagged_for_review():
    v = decide(None, 0.0, error="No tile detected")
    assert v.verdict == REVIEW and v.reason == "No tile detected" and v.value == 0


def test_value_per_tile_and_per_m2():
    assert decide(0.99, 1.0, RuleConfig(price_per_tile=4.0)).value == 4.0
    cfg = RuleConfig(price_basis="per_m2", price_per_m2=40.0, tile_area_m2=0.09)
    assert decide(0.99, 1.0, cfg).value == pytest.approx(3.6)
    assert decide(0.2, 1.0, cfg).value == 0.0


@pytest.mark.parametrize("bad", [{"threshold": 1.2}, {"threshold": 0}, {"review_band": 0.6},
                                 {"price_basis": "per_ton"}, {"price_per_tile": -1}])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        RuleConfig.from_dict(bad)


def test_business_summary():
    s = rules.business_summary({APPROVE: 10, REJECT: 4, "auto_decided": 12}, RuleConfig(price_per_tile=3, cost_bad_tile_shipped=10, cost_manual_check=0.5))
    assert s == {"recovered_value": 30, "errors_avoided": 40, "labour_saved": 6, "unit_value": 3}
