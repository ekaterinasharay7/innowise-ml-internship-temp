import pandas as pd  
from typing import List

# схема разделения данных для валидации и обучения довольно простая: 0-32 месяц обучение, 33 месяц валидация, 34 - тестирование 
class Splitter:
    def __init__(self, time_col, train_end):
        self.time_col = time_col
        self.train_end = train_end

    def split(self, df, feature_cols, target_col):
        train = df[df[self.time_col] < self.train_end]
        val = df[df[self.time_col] == self.train_end]
        X_tr, y_tr = train[feature_cols], train[target_col]
        X_val, y_val = val[feature_cols], val[target_col]
        return X_tr, y_tr, X_val, y_val
