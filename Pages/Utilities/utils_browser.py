from typing import Optional, Union, List, Tuple

# -----------------------------------------------------------------------------
# Sentiment and Risk Helpers
# -----------------------------------------------------------------------------
_SENTIMENT_WEIGHT = {"NEGATIVE": 2, "NEUTRAL": 1, "POSITIVE": 0}


def normalize_label(label: Optional[str]) -> str:
    """Normalize a sentiment label to POSITIVE | NEUTRAL | NEGATIVE."""
    lab = (label or "").upper()
    return lab if lab in {"POSITIVE", "NEGATIVE", "NEUTRAL"} else "NEUTRAL"


def safe_risk(post: dict) -> int:
    """Return the risk score as an integer, defaulting to 0."""
    return int(post.get("risk_score") or 0)


def match_keywords(caption: str, keywords: List[str], require_all: bool) -> bool:
    """Return True if caption matches keywords."""
    if not keywords:
        return True
    cap = (caption or "").lower()
    hits = [k for k in keywords if k.lower() in cap]
    return len(hits) == len(keywords) if require_all else bool(hits)


def sentiment_sort_key(item: dict, source: str) -> Tuple[int, float]:
    """Sort by sentiment weight (NEG > NEU > POS)."""
    if source == "image":
        pred = item.get("prediction") or {}
        lab = normalize_label(pred.get("label"))
        conf = float(pred.get("confidence") or 0.0)
    else:
        lab = normalize_label(item.get("sentiment_label"))
        conf = float(item.get("sentiment_score") or 0.0)
    return (_SENTIMENT_WEIGHT.get(lab, 1), conf)


def passes_sentiment_filter(
    item: dict, mode: str,
    allowed_img: List[str], allowed_txt: List[str]
) -> bool:
    """Return True if a post passes sentiment filters."""
    pred = item.get("prediction") or {}
    img_lab = normalize_label(pred.get("label"))
    txt_lab = normalize_label(item.get("sentiment_label"))

    img_ok = img_lab in allowed_img if allowed_img else True
    txt_ok = txt_lab in allowed_txt if allowed_txt else True

    match mode:
        case "Image only":  return img_ok
        case "Text only":   return txt_ok
        case "Both (AND)":  return img_ok and txt_ok
        case _:             return img_ok or txt_ok


# -----------------------------------------------------------------------------
# CSS Styles
# -----------------------------------------------------------------------------
TILE_CSS = """
<style>
div[data-testid="column"] > div:has(div[data-testid="stVerticalBlock"]) { height: 100%; }
div.tile-card {
  background-color: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 12px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 0 8px rgba(0,0,0,0.25);
}
div.tile-card img { border-radius: 8px; }

.stats-row {
  margin-top: auto;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 6px;
  align-items: center;
}
.badge {
  font-size: 16px;
  line-height: 20px;
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.2);
  flex: 0 1 auto;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.badge.pos { background-color:#21A36633; border-color:#21A366; }
.badge.neu { background-color:#F1C40F33; border-color:#F1C40F; }
.badge.neg { background-color:#E74C3C33; border-color:#E74C3C; }
.badge.low { background-color:#21A36633; border-color:#21A366; }
.badge.medium { background-color:#F1C40F33; border-color:#F1C40F; }
.badge.high { background-color:#E74C3C33; border-color:#E74C3C; }
</style>
"""

# -----------------------------------------------------------------------------
# Mappings
# -----------------------------------------------------------------------------
LAB_MAP = {"POSITIVE": "pos", "NEUTRAL": "neu", "NEGATIVE": "neg"}
RISK_MAP = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}
