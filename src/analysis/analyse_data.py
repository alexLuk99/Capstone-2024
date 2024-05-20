from pathlib import Path
import pandas as pd

from src.analysis.visuals.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.counts_barchart import counts_barchart, normalized_barchart_log, counts_barchart_log
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

    # Extrahiere die gebräuchigsten Wörter aus der Spalter "Fault Description Customer"
    most_common_word(data=df_assistance, output_path=output_path)

    # Barchart mit normalisierte Spalte
    normalized_barchart_log(data=df_assistance, col='VIN', output_path=output_path)

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

    # Michi
    # Michi probiert Nummer 2
    modell = alt.Chart(df_assistance[['Typ aus VIN']]).mark_bar().encode(
        alt.X('Typ aus VIN:N', title='Modelltyp'),
        alt.Y('count(Typ aus VIN):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Typ aus VIN:N', title='Modelltyp'),
            alt.Tooltip('count(Typ aus VIN):Q', title='Anzahl', format=',.0f'),
        ]
    )

    modell.save('output/modell_chart.html')



    # Michis erster Versuch
    model_year_chart = alt.Chart(df_assistance).mark_bar().encode(
        alt.X('Model Year:N', title='Modeljahr'),
        alt.Y('count(Model Year):Q', title='Anzahl'),
        color='Typ aus VIN:N',
        tooltip=[
            alt.Tooltip('Typ aus VIN:N', title='Fahrzeugtyp'),
            alt.Tooltip('Model Year:N', title='Modeljahr'),
            alt.Tooltip('count(Model Year):Q', title='Anzahl', format=',.0f'),
        ]
    )
    model_year_chart.save('output/model_year_chart.html')

    #Michis Versuch Nummer 3
    # Zählen der Anzahl der Vorfälle pro VIN
    incident_counts = df_assistance['VIN'].value_counts().reset_index()
    incident_counts.columns = ['VIN', 'Anzahl Vorfälle']

    # Erstellen des Diagramms
    vin_chart = alt.Chart(incident_counts).mark_bar().encode(
        alt.X('VIN:N', title='VIN-Nummer', sort=alt.EncodingSortField(field='Anzahl Vorfälle', order='descending')),
        alt.Y('Anzahl Vorfälle:Q', title='Anzahl der Vorfälle'),
        tooltip=[
            alt.Tooltip('VIN:N', title='VIN-Nummer'),
            alt.Tooltip('Anzahl Vorfälle:Q', title='Anzahl der Vorfälle', format=',.0f'),
        ]
    ).properties(
        title='Häufigkeit der Anzahl an Vorfällen je VIN-Nummer'
    ).configure_axis(
        labelAngle=-45
    )

    # Speichern des Diagramms als HTML-Datei
    vin_chart.save('output/vin_incident_chart.html')

    # Optional: Ausgabe des Pfads bestätigen
    print("Diagramm gespeichert unter 'output/vin_incident_chart.html'")


    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')
