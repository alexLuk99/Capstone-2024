from typing import List
from loguru import logger
from src.analysis.analyse_data import analyse_data
from src.ml.machine_learning import machine_learning
from src.preprocessing.prepare_data import read_prepare_data
from config.paths import input_path


class Capstone:
    def __init__(self, runners: List, train_cluster_model: bool):
        self.runners = runners
        self.train_cluster_model = train_cluster_model

    def run(self):
        if not input_path.exists():
            logger.error(f'Please save both files to {input_path}')
            exit(-1)
        if 'data_preparation' in self.runners:
            self.load_data()
        if 'analysis' in self.runners:
            self.analyse_data()
        if 'machine_learning' in self.runners:
            self.machine_learning()
        pass

    def load_data(self):
        read_prepare_data()

    def analyse_data(self):
        analyse_data()

    def machine_learning(self):
        machine_learning(train_model=self.train_cluster_model)
