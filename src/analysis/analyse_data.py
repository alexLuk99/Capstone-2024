from pathlib import Path
import pandas as pd

from src.analysis.visuals.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.counts_barchart import counts_barchart, normalized_barchart_log, counts_barchart_log
from src.analysis.visuals.crosstab_heatmap import crosstab_heatmap
from src.analysis.visuals.most_common_word import most_common_word
from src.analysis.visuals.timeline import get_timeline


def analyse_data() -> None:
    # CSV-Datei einlesen
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)
    df_assistance = df_assistance.convert_dtypes()

    # Erstellen des Ausgabeverzeichnisses
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Erstellen von Barcharts
    counts_barchart(data=df_assistance, col='Country Of Origin', output_path=output_path)
    counts_barchart(data=df_assistance, col='Outcome Description', output_path=output_path)
    counts_barchart(data=df_assistance, col='Component', output_path=output_path)

    # Extrahiere die gebräuchigsten Wörter aus der Spalter "Fault Description Customer"
    most_common_word(data=df_assistance, output_path=output_path)

    # Barchart mit normalisierte Spalte
    normalized_barchart_log(data=df_assistance, col='VIN', output_path=output_path)

    # # Häufigkeit der Werte in der Spalte 'Component'
    # component_value_counts = df_assistance['Component'].value_counts()
    # print("Häufigkeit der Werte in der Spalte 'Component':")
    # print(component_value_counts)
    #
    # # Häufigkeit der Werte in der Spalte 'Outcome Description'
    # outcome_value_counts = df_assistance['Outcome Description'].value_counts()
    # print("Häufigkeit der Werte in der Spalte 'Outcome Description':")
    # print(outcome_value_counts)

    # Kreuztabelle erstellen und sortieren
    crosstab_heatmap(data=df_assistance, col1='Component', col2='Outcome Description', output_path=output_path)

    # Aufbereitung Odometer
    counts_barchart_log(data=df_assistance, col='Odometer', output_path=output_path)

    # Jan
    create_country_choropleth(df=df_assistance, column='Country Of Origin', title='Number of permits')
    create_country_choropleth(df=df_assistance, column='Country Of Incident', title='Number of incidents')

    perform_chi_square_test(data=df_assistance, col1='Component', col2='Outcome Description')
    perform_chi_square_test(data=df_assistance, col1='Component', col2='Reason Of Call')
    perform_chi_square_test(data=df_assistance, col1='Outcome Description', col2='Reason Of Call')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Component')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Outcome Description')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Reason Of Call')

    get_timeline(data=df_assistance, col='Incident Date', aggregate='Monat')
    # Speichern der Visualisierung als HTML

    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')
