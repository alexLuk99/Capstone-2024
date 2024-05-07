import pandas as pd

from src.analysis.prepare_data import read_prepare_data


class Capstone:
    def __init__(self):
        self.data = None

    def load_data(self):
        self.data = read_prepare_data()
