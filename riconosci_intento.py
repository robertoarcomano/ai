from transformers import pipeline

# Creiamo una pipeline di "zero-shot-classification" per riconoscere intenti
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def riconosci_intento(testo_utente):
    candidate_labels = ["saldo conto", "operazioni", "addebiti", "accrediti"]
    risultato = classifier(testo_utente, candidate_labels)
    return risultato['labels'][0]  # etichetta con maggiore confidenza

# Test su richiesta utente
testo = "bonifici in entrata"
print(riconosci_intento(testo))  # restituisce "saldo conto"

