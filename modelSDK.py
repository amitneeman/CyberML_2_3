from bert_sklearn import load_model
import pandas as pd

MODEL_PATH = './bertmodel/mymodel.bin'


class ModelSKD:
    def __init__(self, bert_model):
        self.bert_model = bert_model

    @classmethod
    def load_model(cls, model_file=MODEL_PATH):
        print("Before model load")
        model = load_model(model_file)
        print("after model load")
        return cls(model)

    def predict_label_for_chat(self, chat) -> str:
        data = pd.DataFrame([{"segment": chat["chat"]}])
        res = self.bert_model.predict(data["segment"])
        return res[0]
