from typing import List

from src.preprocessing.prepare_data import read_prepare_data


class Capstone:
    def __init__(self, runners: List):
        self.data = None
        self.runners = runners

    def run(self):
        if 'data_preparation' in self.runners:
            self.load_data()
        if 'analysis' in self.runners:
            self.analyse_data()
        pass

    def load_data(self):
        self.data = read_prepare_data()

    def analyse_data(self):
        pass
