import re
import emoji
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

_sentiment_pipeline = None

def preprocess_social_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = emoji.demojize(text)       # keep emoji meaning
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#", "", text)     # keep hashtag words
    text = re.sub(r"\s+", " ", text)  # clean extra spaces
    return text.strip()


def analyze_sentiment(text: str):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        # Load model directly from Hugging Face (online)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        _sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    clean_text = preprocess_social_text(text)
    if not clean_text:
        return {"label": "NEUTRAL", "score": 0.0}

    result = _sentiment_pipeline(clean_text[:512])[0]

    # SST-2 labels are 'POSITIVE' and 'NEGATIVE'
    label_mapping = {"NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"}
    label = label_mapping.get(result["label"], "NEUTRAL")

    return {"label": label, "score": round(result["score"], 3)}
