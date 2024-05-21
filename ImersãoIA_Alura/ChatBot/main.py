import google.generativeai as genai
from decouple import config

TOKEN = config("SECRET_TOKEN")
genai.configure(api_key=TOKEN)

config = {
    "candidate_count": 1, #Quantidade de Respostas, pode vir mais de uma
    "temperature": 0.5, #Aleatoriedade de Palavras
}
safe_config = {
    "HARASSMENT": "BLOCK_LOW_AND_ABOVE", #Bloquear Assedio 
    "HATE": "BLOCK_LOW_AND_ABOVE", #Bloquear Discurso de Odio
    "SEXUAL": "BLOCK_LOW_AND_ABOVE", #Bloquear Conteudo Sexual
    "DANGEROUS": "BLOCK_LOW_AND_ABOVE", #Bloquear Conteudo Perigoso
}

modelo = genai.GenerativeModel(model_name='gemini-1.0-pro', generation_config=config, safety_settings=safe_config)
chat = modelo.start_chat(history=[])

prompt = ""
print("Comece a conversa...")
while prompt.lower() != "sair":
    prompt = input()
    resposta = chat.send_message(prompt)
    print("-",resposta.text)