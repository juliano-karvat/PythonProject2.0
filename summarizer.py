from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Resumo(BaseModel):
    titulo: str
    resumo: str
    palavras_chave: list[str]


def gerar_resumo(texto: str) -> Resumo:
    prompt = f"""
    Resuma o seguinte texto e retorne em formato JSON com:
    - titulo
    - resumo (até 100 palavras)
    - palavras_chave (5 itens)

    Texto:
    {texto}
    """

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    conteudo = resposta.choices[0].message.content


    conteudo = conteudo.strip()
    if conteudo.startswith("```"):
        conteudo = conteudo.split("\n", 1)[1]  # Remove a primeira linha com ```
        if conteudo.endswith("```"):
            conteudo = conteudo.rsplit("\n", 1)[0]  # Remove a última linha com ```
    conteudo = conteudo.strip()


    dados = json.loads(conteudo)

    return Resumo.model_validate(dados)


if __name__ == "__main__":
    texto_longo = """
    A inteligência artificial (IA) é um campo da ciência da computação 
    que busca criar sistemas capazes de realizar tarefas 
    que normalmente requerem inteligência humana...
    """

    resultado = gerar_resumo(texto_longo)
    print(json.dumps(resultado.model_dump(), indent=2, ensure_ascii=False))