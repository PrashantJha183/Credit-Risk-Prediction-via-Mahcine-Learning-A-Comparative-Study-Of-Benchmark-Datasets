import pandas as pd
import os

def load_uci(path="data/default_of_credit_card_clients.xls"):
    """
    Loads the UCI Credit Card Default dataset from an Excel file.
    Assumes the actual data starts from the second row (header=1).
    """
    return pd.read_excel(path, header=1)

def load_german(path="data/south_german_credit.csv"):
    """
    Loads the South German Credit dataset from a CSV file.
    """
    return pd.read_csv(path)
