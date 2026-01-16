# MachineInnovators Inc. - Sentiment Analysis MLOps

Questo progetto implementa una pipeline scalabile per l'analisi del sentiment sui social media, sviluppata per **MachineInnovators Inc.** L'obiettivo è automatizzare il monitoraggio della reputazione online utilizzando metodologie MLOps moderne e modelli Transformer.

Obiettivo del Progetto
Monitorare la reputazione aziendale classificando automaticamente i feedback degli utenti in tre categorie:
 Positive
 Neutral
 Negative

Il sistema è progettato per abilitare un intervento rapido del team di supporto in caso di sentiment negativo.

Scelta del Modello: Transformer vs FastText
Sebbene inizialmente fosse stato valutato l'uso di FastText, la scelta finale è ricaduta su RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
Motivazione:I modelli basati su architettura Transformer (come BERT/RoBERTa) offrono una comprensione del contesto superiore rispetto agli approcci statistici classici, gestendo meglio lo slang tipico di Twitter e le sfumature di significato.

Limitazioni Note (Analisi dei Risultati)
Durante la fase di testing, è emersa una forte dipendenza del modello dalla lingua inglese.
* Input in Inglese (es. *"Terrible service"*) -> Classificazione corretta (Negative).
* Input in Italiano (es. *"Pessimo servizio"*) -> Classificazione errata o incerta Neutral).


Struttura del Repository
* `app.py`: Logica principale dell'applicazione e classe `SentimentAnalyzer`.
* `test_app.py`: Unit tests per la verifica automatica della pipeline.
* `.github/workflows`: Pipeline CI/CD configurata con GitHub Actions.
* `requirements.txt`: Elenco delle dipendenze Python.

Pipeline CI/CD
Il progetto integra una pipeline di Continuous Integration che:
1.  Si attiva ad ogni `push` sul branch principale.
2.  Installa l'ambiente e le dipendenze.
3.  Esegue i test automatici per garantire che il modello risponda correttamente prima del deploy.

Come eseguire in locale
```bash
pip install -r requirements.txt
python app.py
