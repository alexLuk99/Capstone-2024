import pandas as pd
from config.paths import interim_path
from src.ml.models.clustering import clustering


def machine_learning() -> None:
    df_assistance = pd.read_csv(interim_path / 'assistance.csv')
    df_workshop = pd.read_csv(interim_path / 'workshop.csv')
    df_merged = pd.read_csv(interim_path / 'merged.csv')

    # classification with leads

    # clustering
    clustering(df_assistance=df_assistance, df_workshop=df_workshop, df_merged=df_merged)