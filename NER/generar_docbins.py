import spacy
from spacy.tokens import DocBin
import json
from spacy.util import filter_spans

def entrenar(data, filename: str):
    nlp = spacy.blank("es")

    db = DocBin()

    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations["entities"]:
            
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                print(msg)
            else:
                ents.append(span)

        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        db.add(doc)
    db.to_disk("./"+filename+".spacy")


def particionar(train, test):
    train=train["annotations"]
    test=test["annotations"]
    entrenar(train, "train_data")
    entrenar(test, "test_data")

train_data= json.load(open("anotaciones.json", 'r', encoding='utf-8'))
test_data= json.load(open("testing.json", 'r', encoding='utf-8'))

particionar(train_data, test_data)



