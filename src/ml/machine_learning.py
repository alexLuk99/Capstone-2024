import pandas as pd
from config.paths import interim_path
from src.ml.models.analyze_clusters import analyze_clusters
from src.ml.models.classification import classification
from src.ml.models.classification_suspect import classification_suspect
from src.ml.models.clustering import clustering
from src.ml.prepare_for_ml import prepare_for_ml


def machine_learning(train_model: bool) -> None:
    df_assistance = pd.read_csv(interim_path / 'assistance.csv')
    df_workshop = pd.read_csv(interim_path / 'workshop.csv')
    df_merged = pd.read_csv(interim_path / 'merged.csv')

    # classification with leads

    # clustering
    data, data_suspect = prepare_for_ml(df_assistance=df_assistance, df_workshop=df_workshop)

    # clustering(data=data, train_model=train_model)
    # analyze_clusters()

    classification_suspect(data=data_suspect, train_model=train_model)

    # categorisation
    classification(df_assistance=df_assistance, df_workshop=df_workshop)