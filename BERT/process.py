
import pandas as pd
import spacy 
from transformers import pipeline

nlp = spacy.load('es_core_news_lg')

pipe = pipeline("question-answering", model="rvargas93/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", handle_impossible_answer=True)

preguntas= [
    "¿Cuál es la dirección del lote?",
    "¿Cuál es el valor del FOT?",
    "¿El terreno es irregular?",
    "¿Cuáles son las dimensiones del lote?",
    "¿El lote está en una esquina?",
    "¿En qué barrio privado está ubicado?",
    "¿Cuántos frentes tiene el inmueble?",
    "¿El inmueble tiene piscina?"
]

input = pd.read_csv('ground_truth_75.csv', sep = '|')
input = input.fillna("")
metricas = {
    "direccion": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "fot": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "irregular": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "medidas": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "esquina": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "barrio": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "frentes": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    },
    "pileta": {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "error": [
            
        ]
    }
}

for index, row in input.iterrows():
    respuestas= pipe(question=preguntas, context=row["descripcion"], handle_impossible_answer=True)
    for respuesta, esperada, key_metrica in zip(respuestas, list(row[1:]), metricas):
        respuesta= respuesta["answer"].strip()
        respuesta= ' '.join([token.text for token in nlp(respuesta) if not token.is_punct])
        if respuesta == "" and esperada == "":
            metricas[key_metrica]["tn"]+=1
        else:
            if key_metrica in ["direccion","fot", "medidas", "barrio", "frentes"]:
                correcta= nlp(respuesta).similarity(nlp(esperada)) > 0.9
            elif key_metrica in [ "irregular", "esquina", "pileta"]:
                correcta= respuesta != "" and esperada == True

            if correcta:
                metricas[key_metrica]["tp"]+=1
            else:
                metricas[key_metrica]["error"].append({
                    "contexto": row["descripcion"],
                    "respuesta_predicha": respuesta,
                    "respuesta_esperada": esperada
                })
                if respuesta == "" and esperada != "":
                    metricas[key_metrica]["fn"]+=1
                elif (esperada == "" and respuesta != "") or (esperada != respuesta):
                    metricas[key_metrica]["fp"]+=1
            

for metrica, valores in metricas.items():
    tp = valores["tp"]
    fp = valores["fp"]
    fn = valores["fn"]

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    metricas[metrica]["p"] = precision
    metricas[metrica]["r"] = recall
    metricas[metrica]["f1"] = f1_score

import json
with open('rvargas93.json', 'w', encoding="utf8") as fp:
    json.dump(metricas, fp, ensure_ascii=False)
            
            