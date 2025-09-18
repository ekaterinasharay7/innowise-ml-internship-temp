import pandas as pd


class ExpandingWindowSplitter:
    def __init__(self, time_col, first_val_month, last_val_month):
        self.time_col = time_col
        self.first_val_month = first_val_month
        self.last_val_month = last_val_month

    def split(self, df):
        for val_month in range(self.first_val_month, self.last_val_month + 1):
            df_tr = df[df[self.time_col] < val_month]
            df_val = df[df[self.time_col] == val_month]
            yield val_month, df_tr, df_val
