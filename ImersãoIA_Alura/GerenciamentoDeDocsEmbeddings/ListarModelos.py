import google.generativeai as genai
from decouple import config

TOKEN = config("SECRET_TOKEN")
genai.configure(api_key=TOKEN)
modelo = genai.GenerativeModel('gemini-pro')

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)