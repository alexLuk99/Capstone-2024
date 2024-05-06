import pandas as pd

class Capstone:
    def __init__(self):
        self.data = None

    def load_data(self):
        self.data = read_prepare_data()