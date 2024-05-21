import google.generativeai as genai
import pandas as pd
import numpy as np
from decouple import config

TOKEN = config("SECRET_TOKEN")
genai.configure(api_key=TOKEN)
modelo = "models/embedding-001"

doc1 = {
    "Titulo": "O que é uma Maça?",
    "Conteudo": "A maçã é o pseudofruto pomáceo da macieira, árvore da família Rosaceae. É um dos pseudofrutos de árvore mais cultivados, e o mais conhecido dos muitos membros do género Malus que são usados ​​pelos seres humanos."
}
doc2 = {
    "Titulo": "O que é uma Pera?",
    "Conteudo": "A pera é o fruto comestível da pereira, uma árvore do gênero Pyrus L., pertencente à família Rosaceae, e que conta com vinte variedades de espécies cultivadas em todo o mundo. A fruta pode ser consumida in natura, enlatada, em suco e desidratada."
}
doc3 = {
    "Titulo": "O que é um texto em português?",
    "Conteudo": "Um texto é uma manifestação da linguagem. Pode ser definido como tudo aquilo que é dito por um emissor e interpretado por um receptor. Dessa forma, tudo que é interpretável é um texto. Outra forma de conceituação é pensar que tudo aquilo que produz um sentido completo, que seja uma mensagem compreensível, é um texto."
}
documentos = [doc1, doc2, doc3]
df = pd.DataFrame(documentos)
#df.columns = ["Título", "Contúdo"]

def embed(title, text):
    return genai.embed_content(model=modelo, content=text, title=title, task_type="RETRIEVAL_DOCUMENT")["embedding"]

df["Embeddings"] = df.apply(lambda row: embed(row["Titulo"], row["Conteudo"]), axis=1)
print(df)

def GerarEConsultar(consulta, base):
    embedDaConsulta = genai.embed_content(model=modelo, content=consulta, task_type="RETRIEVAL_QUERY")["embedding"]
    ProdutosEscalares = np.dot(np.stack(df["Embeddings"]), embedDaConsulta)
    indice = np.argmax(ProdutosEscalares)
    return df.iloc[indice]["Conteudo"]

consulta = "O que define uma maça?"

#Controle contra Alucinações
trecho = GerarEConsultar(consulta, df)
print("\n",trecho,"\n")

#Aplicando a Genai 
prompt = f"Reescreva esse texto de uma forma mais descrontraida, sem adicionar informações que não fazem parte do texto: {trecho}"
modelo2 = genai.GenerativeModel("gemini-1.0-pro")
resposta = modelo2.generate_content(prompt)
print(resposta. text)