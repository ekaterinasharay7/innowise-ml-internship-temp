import pandas as pd  
from typing import List


class Validator:
    def __init__(self, df, splitter, model, metric):
        self.df = df
        self.splitter = splitter
        self.model = model
        self.metric = metric

    def run(self, feature_cols: list, target_col: str):
        X_tr, y_tr, X_val, y_val = self.splitter.split(self.df, feature_cols, target_col)
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_val)
        score = self.metric(y_val, y_pred)
        return {"val_score": score, "pred_val": y_pred}
