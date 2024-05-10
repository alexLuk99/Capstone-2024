from typing import List

import pandas as pd

from src.analysis.prepare_data import read_prepare_data


class Capstone:
    def __init__(self, runners: List):
        self.data = None
        self.runners = runners

    def run(self):
        # do something meaningful
        pass


    def load_data(self):
        self.data = read_prepare_data()
