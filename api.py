from openai import OpenAI
import os

def load_api_key():
    with open(os.path.expanduser("~/.perplexity_api_key"), "r") as file:
        return file.read().strip()

client = OpenAI(
    api_key=load_api_key(),
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar-pro",
    messages=[
        {"role": "system", "content": "Rispondi solo alla domanda in modo conciso, senza spiegazioni aggiuntive e non ripetere la domanda, dimmi solo la risposta, senza citare le fonti o mettere asterischi. Quando rispondi non includere mai la domanda, dai solo la risposta finale"},
        {"role": "user", "content": "0/0"}
    ]
)

print(response.choices[0].message.content)

