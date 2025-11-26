from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Carica il tokenizer e il modello BERT pre-addestrato per la classificazione di sequenze
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Crea una pipeline per la sentiment analysis
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Testo da analizzare
testo = "Questo prodotto è davvero eccellente e lo consiglio a tutti!"
testo = "Questo prodotto non é buono"

# Esegui la predizione
risultato = nlp(testo)

print(risultato)
