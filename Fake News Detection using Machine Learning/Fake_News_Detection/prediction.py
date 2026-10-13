import os
import joblib
import numpy as np
from preprocessing import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

_model = None
_vectorizer = None


def load_model_and_vectorizer():
    """
    Loads the joblib model and vectorizer binaries with error handling.
    """
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer file not found at {VECTORIZER_PATH}. Please run train_model.py first.")
        
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        
    return _model, _vectorizer


def get_explainable_keywords(processed_text: str, top_n: int = 5):
    """
    Extracts top keywords contributing to the prediction based on TF-IDF features
    and Logistic Regression weights.
    
    Args:
        processed_text (str): Preprocessed text string.
        top_n (int): Number of top terms to return.
        
    Returns:
        dict: Top fake-leaning and top real-leaning words detected in input.
    """
    model, vectorizer = load_model_and_vectorizer()
    
    tokens = list(set(processed_text.split()))
    if not tokens:
        return {"fake_indicators": [], "real_indicators": []}
    
    feature_names = vectorizer.get_feature_names_out()
    feature_index_map = {feat: idx for idx, feat in enumerate(feature_names)}
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
    else:
        return {"fake_indicators": [], "real_indicators": []}
        
    fake_terms = []
    real_terms = []
    
    for token in tokens:
        if token in feature_index_map:
            idx = feature_index_map[token]
            weight = coefs[idx]
            if weight > 0:
                real_terms.append((token, weight))
            else:
                fake_terms.append((token, abs(weight)))
                
    real_terms.sort(key=lambda x: x[1], reverse=True)
    fake_terms.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "fake_indicators": [term[0] for term in fake_terms[:top_n]],
        "real_indicators": [term[0] for term in real_terms[:top_n]]
    }


def predict_news(headline: str = "", article_text: str = "") -> dict:
    """
    Predicts whether a news article is REAL or FAKE.
    
    Args:
        headline (str): Headline of the news article.
        article_text (str): Body content of the news article.
        
    Returns:
        dict: Result dictionary containing prediction label, confidence score,
              color theme, status code, and top word indicators.
    """
    headline_clean = headline.strip() if headline else ""
    article_clean = article_text.strip() if article_text else ""
    
    if not headline_clean and not article_clean:
        return {
            "status": "error",
            "message": "Please enter a news headline or article body text to analyze.",
            "label": "INVALID INPUT",
            "confidence": 0.0,
            "color": "#6c757d"
        }
        
    combined_raw = f"{headline_clean} {article_clean}"
    
    processed = preprocess_text(combined_raw)
    
    if not processed or len(processed.split()) == 0:
        return {
            "status": "error",
            "message": "Input contains only stopwords or invalid characters. Please provide meaningful text.",
            "label": "UNPROCESSABLE INPUT",
            "confidence": 0.0,
            "color": "#6c757d"
        }
        
    try:
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load machine learning model: {str(e)}",
            "label": "SYSTEM ERROR",
            "confidence": 0.0,
            "color": "#6c757d"
        }
        
    tfidf_vec = vectorizer.transform([processed])
    
    pred = model.predict(tfidf_vec)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(tfidf_vec)[0]
        confidence = float(np.max(probabilities) * 100)
        real_prob = float(probabilities[1] * 100)
        fake_prob = float(probabilities[0] * 100)
    else:
        confidence = 100.0
        real_prob = 100.0 if pred == 1 else 0.0
        fake_prob = 100.0 if pred == 0 else 0.0

    label = "REAL NEWS" if pred == 1 else "FAKE NEWS"
    color = "#28a745" if pred == 1 else "#dc3545"  # Green = Real, Red = Fake
    
    explainability = get_explainable_keywords(processed)
    
    return {
        "status": "success",
        "label": label,
        "is_real": bool(pred == 1),
        "confidence": round(confidence, 2),
        "real_probability": round(real_prob, 2),
        "fake_probability": round(fake_prob, 2),
        "color": color,
        "preprocessed_snippet": processed[:100] + "..." if len(processed) > 100 else processed,
        "indicators": explainability
    }


if __name__ == "__main__":
    test_fake = predict_news(
        headline="BOMBSHELL: Hidden Alien Artifact Discovered Beneath Antarctic Ice Sheet!",
        article_text="Anonymous whistleblower claims top secret military forces uncovered alien technology."
    )
    print("Fake Sample Test Result:", test_fake)

    test_real = predict_news(
        headline="Senate Passes Bipartisan Infrastructure Bill After Weeks of Debate",
        article_text="WASHINGTON (Reuters) - Official representatives confirmed today that the legislation was approved."
    )
    print("Real Sample Test Result:", test_real)
