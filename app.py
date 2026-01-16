import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Inizializza il modello e il tokenizer.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Caricamento modello {model_name} su {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.labels = ['Negative', 'Neutral', 'Positive']

    def predict(self, text):
        """
        Riceve un testo e restituisce il sentiment e lo score di confidenza.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded_input)

        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        # Trova la classe con il punteggio più alto
        ranking = np.argsort(scores)[::-1]
        top_label = self.labels[ranking[0]]
        confidence = scores[ranking[0]]

        return {
            "sentiment": top_label,
            "confidence": float(confidence),
            "scores": {self.labels[i]: float(scores[i]) for i in range(len(self.labels))}
        }

# Blocco per testare lo script se eseguito direttamente
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_text = "MLOps is amazing for continuous monitoring!"
    print(f"Testo: {test_text}")
    print(f"Risultato: {analyzer.predict(test_text)}")
