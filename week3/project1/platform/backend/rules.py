"""
All verdict and money logic lives here, and nowhere else.

Decision rule (confirmed): usable-area threshold.
    usable_fraction >= threshold + band   -> APPROVE
    usable_fraction <  threshold - band   -> REJECT
    otherwise                             -> REVIEW   (a human looks at it)
A tile also goes to REVIEW when the model is unsure about its own mask
(confidence below `min_confidence`) or when no tile was found in the photo.

usable_fraction = usable tile pixels / all tile pixels (usable + damaged zone).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

APPROVE, REVIEW, REJECT = "APPROVE", "REVIEW", "REJECT"
VERDICTS = (APPROVE, REVIEW, REJECT)


@dataclass
class RuleConfig:
    threshold: float = 0.85          # APPROVE if usable share >= this
    review_band: float = 0.05        # +/- band around the threshold that goes to a human
    min_confidence: float = 0.40     # below this the tile goes to REVIEW whatever its usable share
    # money
    price_basis: str = "per_tile"    # per_tile | per_m2
    price_per_tile: float = 3.20     # EUR a sellable reclaimed tile fetches
    price_per_m2: float = 38.0       # EUR per m2 of sellable tile
    tile_area_m2: float = 0.09       # nominal tile size for the m2 price (30 x 30 cm)
    cost_bad_tile_shipped: float = 14.0   # EUR: return, re-delivery and goodwill for a defective tile sold
    cost_manual_check: float = 0.45       # EUR: inspector time to check one tile by hand
    currency: str = "EUR"

    @classmethod
    def from_dict(cls, d: dict | None) -> "RuleConfig":
        d = d or {}
        names = {f.name: f for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in names and v is not None:
                kw[k] = str(v) if names[k].type == "str" else float(v)
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def validate(self):
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        if not 0.0 <= self.review_band < 0.5:
            raise ValueError("review_band must be between 0 and 0.5")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.price_basis not in ("per_tile", "per_m2"):
            raise ValueError("price_basis must be 'per_tile' or 'per_m2'")
        for k in ("price_per_tile", "price_per_m2", "tile_area_m2", "cost_bad_tile_shipped", "cost_manual_check"):
            if getattr(self, k) < 0:
                raise ValueError(f"{k} must not be negative")

    def to_dict(self):
        return asdict(self)


@dataclass
class Verdict:
    verdict: str
    reason: str
    confidence: float          # 0..1, combined
    decision_margin: float     # signed usable_fraction - threshold
    margin_score: float        # 0..1, how far from the threshold (in review bands)
    mask_certainty: float      # 0..1, from the model
    value: float = 0.0         # EUR recovered if the tile is sold

    def to_dict(self):
        return asdict(self)


def unit_value(cfg: RuleConfig) -> float:
    """Sale value of one sellable tile."""
    if cfg.price_basis == "per_m2":
        return cfg.price_per_m2 * cfg.tile_area_m2
    return cfg.price_per_tile


def value_for(verdict: str, cfg: RuleConfig) -> float:
    return round(unit_value(cfg), 4) if verdict == APPROVE else 0.0


def margin_score(usable_fraction: float, cfg: RuleConfig) -> float:
    """0 on the threshold, 0.5 at the edge of the review band, 1 at two bands (or more) away."""
    scale = 2 * max(cfg.review_band, 0.01)
    return float(min(1.0, abs(usable_fraction - cfg.threshold) / scale))


def combine_confidence(margin: float, certainty: float) -> float:
    """Geometric mean: a tile is only 'sure' if it is far from the line AND the mask is crisp."""
    return float(math.sqrt(max(0.0, margin) * max(0.0, min(1.0, certainty))))


def decide(usable_fraction: float | None, mask_certainty: float = 1.0, cfg: RuleConfig | None = None,
           error: str | None = None) -> Verdict:
    cfg = cfg or RuleConfig()
    if error or usable_fraction is None:
        return Verdict(REVIEW, error or "No tile detected", 0.0, 0.0, 0.0, float(mask_certainty), 0.0)

    uf = float(usable_fraction)
    m = margin_score(uf, cfg)
    conf = combine_confidence(m, mask_certainty)
    lo, hi = round(cfg.threshold - cfg.review_band, 6), round(cfg.threshold + cfg.review_band, 6)
    pct = lambda x: f"{100 * x:.0f}%"

    if cfg.review_band > 0 and lo <= uf < hi:
        v, why = REVIEW, f"Usable {pct(uf)} is within ±{pct(cfg.review_band)} of the {pct(cfg.threshold)} line"
    elif uf >= cfg.threshold:
        v, why = APPROVE, f"Usable {pct(uf)} ≥ {pct(cfg.threshold)}"
    else:
        v, why = REJECT, f"Usable {pct(uf)} < {pct(cfg.threshold)}"

    if v != REVIEW and conf < cfg.min_confidence:
        v, why = REVIEW, f"Low confidence ({pct(conf)}); model would say {'approve' if uf >= cfg.threshold else 'reject'}"

    return Verdict(v, why, round(conf, 4), round(uf - cfg.threshold, 4), round(m, 4),
                   round(float(mask_certainty), 4), value_for(v, cfg))


def business_summary(counts: dict, cfg: RuleConfig) -> dict:
    """
    Money view of a set of final verdicts.
        recovered_value   approved tiles x sale value
        errors_avoided    rejected tiles that would otherwise have been shipped x cost of a bad tile
        labour_saved      tiles decided without a human x cost of a manual check
    """
    approved = counts.get(APPROVE, 0)
    rejected = counts.get(REJECT, 0)
    auto = counts.get("auto_decided", approved + rejected)
    return {
        "recovered_value": round(approved * unit_value(cfg), 2),
        "errors_avoided": round(rejected * cfg.cost_bad_tile_shipped, 2),
        "labour_saved": round(auto * cfg.cost_manual_check, 2),
        "unit_value": round(unit_value(cfg), 2),
    }
