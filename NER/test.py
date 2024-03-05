import pandas as pd
import spacy


nlp= spacy.load("output-merge/model-best")


metricas = {
    "DIRECCION": {
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
    "FOT": {
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
    "IRREGULAR": {
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
    "DIMENSIONES": {
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
    "ESQUINA": {
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
    "NOMBRE_BARRIO": {
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
    "CANT_FRENTES": {
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
    "PILETA": {
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

input = pd.read_csv('ground_truth_75.csv', sep = '|')
input = input.fillna("")

for index, row in input.iterrows():
   respuestas= {
       "DIRECCION": "",
       "FOT": "",
       "IRREGULAR":"",
       "DIMENSIONES": "",
       "ESQUINA": "",
       "NOMBRE_BARRIO":"",
       "CANT_FRENTES": "",
       "PILETA": ""
   }
   doc=nlp(row['descripcion'])
   for ent in doc.ents:
        respuestas[ent.label_]+= ent.text+" "
   for respuesta, esperada, key_metrica in zip(respuestas.values(), list(row[1:]), metricas):
        if respuesta == "" and esperada == "":
            metricas[key_metrica]["tn"]+=1
        else:
            if key_metrica in ["CANT_FRENTES","DIMENSIONES", "DIRECCION", "FOT", "NOMBRE_BARRIO"]:
                correcta= nlp(respuesta).similarity(nlp(esperada)) > 0.9
            elif key_metrica in [ "ESQUINA", "IRREGULAR", "PILETA"]:
                correcta= True if esperada==respuestas[key_metrica] or (respuestas[key_metrica] != "" and esperada == True) else False
            
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
                elif (esperada == "" and respuesta != "") or (esperada!=respuesta):
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
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1_score = 0.0

    metricas[metrica]["p"] = precision
    metricas[metrica]["r"] = recall
    metricas[metrica]["f1"] = f1_score

import json
with open('resultados.json', 'w', encoding="utf8") as fp:
    json.dump(metricas, fp, ensure_ascii=False)