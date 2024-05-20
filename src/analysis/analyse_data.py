from pathlib import Path
import pandas as pd

from src.analysis.visuals.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.component_outcome_services import component_outcome_services
from src.analysis.visuals.counts_barchart import counts_barchart, normalized_barchart_log, counts_barchart_log, \
    counts_barchart_color
from src.analysis.visuals.crosstab_heatmap import crosstab_heatmap
from src.analysis.visuals.most_common_word import most_common_word
from src.analysis.visuals.start_end_date import policy_start_end_date
from src.analysis.visuals.timeline import get_timeline


def analyse_data() -> None:
    # CSV-Datei einlesen
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)
    df_assistance = df_assistance.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Difference between policy start/end date with first assistance call
    policy_start_end_date(data=df_assistance, output_path=output_path)

    # Erstellen von Barcharts
    counts_barchart(data=df_assistance, col='Country Of Origin', output_path=output_path)
    counts_barchart(data=df_assistance, col='Outcome Description', output_path=output_path)
    counts_barchart(data=df_assistance, col='Component', output_path=output_path)
    counts_barchart(data=df_assistance, col='Typ aus VIN', output_path=output_path)
    # counts_barchart(data=df_assistance, col='VIN', output_path=output_path)

    # Erstellen von Barcharts mit Farbe
    counts_barchart_color(data=df_assistance, col='Model Year', color='Typ aus VIN', output_path=output_path)
    counts_barchart_color(data=df_assistance, col='Component', color='Outcome Description', output_path=output_path)
    counts_barchart_color(data=df_assistance, col='Model Year', color='Typ aus VIN', output_path=output_path)

    # Extrahiere die gebräuchigsten Wörter aus der Spalter "Fault Description Customer"
    most_common_word(data=df_assistance, output_path=output_path)

    # Barchart mit normalisierte Spalte
    normalized_barchart_log(data=df_assistance, col='VIN', output_path=output_path)

    # Kreuztabelle erstellen und sortieren
    crosstab_heatmap(data=df_assistance, col1='Component', col2='Outcome Description', output_path=output_path)

    # Odometer
    counts_barchart_log(data=df_assistance, col='Odometer', output_path=output_path)

    # Abhängigkeit zwischen Component, Outcome und Services darstellen
    component_outcome_services(data=df_assistance, output_path=output_path)

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

    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
