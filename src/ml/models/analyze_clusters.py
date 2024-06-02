import pandas as pd

from config.paths import input_path


def analyze_clusters():
    cluster = pd.read_csv(input_path / 'clustered.csv')

    # Visualisierungen um Cluster zu analysieren
