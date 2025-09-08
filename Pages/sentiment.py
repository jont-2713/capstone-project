from transformers import pipeline

_sentiment_pipeline = None

def analyze_sentiment(text: str):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = _sentiment_pipeline(text[:512])[0]
    return {"label": result["label"], "score": round(result["score"], 3)}
