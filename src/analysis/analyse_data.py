from pathlib import Path
import pandas as pd
import altair as alt

def analyse_data() -> None:
    df_assistance = pd.read_csv('data/interim/assistance.csv')
    df_assistance = df_assistance.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Example 1
    country_of_origin = alt.Chart(df_assistance).mark_bar().encode(
        alt.X('Country Of Origin:N', title='Unfallsland'),
        alt.Y('count(Country Of Origin):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Country Of Origin:N', title='Unfallsland'),
            alt.Tooltip('count(Country Of Origin):Q', title='Anzahl', format=',.0f'),
        ]
    )
    country_of_origin.save('output/country_of_origin_chart.html')

    # Example 2
    component2 = alt.Chart(df_assistance[['Component']]).mark_bar().encode(
        alt.X('Component:N', title='Komponente'),
        alt.Y('count(Component):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Component:N', title='Komponente'),
            alt.Tooltip('count(Component):Q', title='Anzahl', format=',.0f'),
        ]
    )

    component2.save('output/component_chart3.html')

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


    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
