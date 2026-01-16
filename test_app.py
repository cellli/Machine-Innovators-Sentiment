import unittest
from app import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Inizializziamo il modello una volta sola per tutti i test
        print("Setup test: Caricamento modello...")
        cls.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        # Testiamo se una frase chiaramente positiva viene riconosciuta come tale
        text = "I absolutely love this new feature, it is fantastic!"
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'Positive')

    def test_negative_sentiment(self):
        # Testiamo se una frase negativa viene riconosciuta
        text = "This is the worst experience I have ever had. Terrible."
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'Negative')

if __name__ == '__main__':
    unittest.main()
