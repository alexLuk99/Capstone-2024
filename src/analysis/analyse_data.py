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
    country_of_orgigin = alt.Chart(df_assistance).mark_bar().encode(
        alt.X('Country Of Origin:N', title='Unfallsland'),
        alt.Y('count(Country Of Origin):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Country Of Origin:N', title='Unfallsland'),
            alt.Tooltip('count(Country Of Origin):Q', title='Anzahl', format=',.0f'),
        ]
    )
    country_of_orgigin.save('output/country_of_orgigin_chart.html')

    # Example 2
    component = alt.Chart(df_assistance[['Component']]).mark_bar().encode(
        alt.X('Component:N', title='Komponente'),
        alt.Y('count(Component):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Component:N', title='Komponente'),
            alt.Tooltip('count(Component):Q', title='Anzahl', format=',.0f'),
        ]
    )

    # Example Tim 1
    # Häufigkeiten berechnen
    freq_df = df_assistance.groupby(['Component', 'Outcome Description']).size().reset_index(name='Count')

    # Erstellen eines gestapelten Balkendiagramms
    chart = alt.Chart(freq_df).mark_bar().encode(
        x='Component:N',
        y='Count:Q',
        color='Outcome Description:N'
    ).properties(
        title='Components vs Outcome'
    )

    # Ausgabe als HTML
    chart.save('components_vs_outcome.html')

    print("Chart saved as 'components_vs_outcome.html'")

    df_workshop = pd.read_csv('data/interim/workshop.csv')


    # Example Tim 2

    # Daten vorbereiten: Melt-Transformation für Services
    melted_df = df_assistance.melt(id_vars=['Component', 'Outcome Description'], value_vars=['Hotel Service', 'Alternative Transport', 'Taxi Service', 'Vehicle Transport', 'Car Key Service', 'Parts Service'],
                        var_name='Service', value_name='Value')

    # Häufigkeiten berechnen
    freq_df = melted_df.groupby(['Component', 'Outcome Description', 'Service', 'Value']).size().reset_index(name='Count')

    # Erstellen eines gestapelten Balkendiagramms
    chart = alt.Chart(freq_df).mark_bar().encode(
        x='Component:N',
        y='Count:Q',
        color='Value:N'
    ).facet(
        column=alt.Column('Service:N', header=alt.Header(labelAngle=-90, title='Service'))
    ).properties(
        title='Influence of Components and Outcome on Services'
    )

    # Ausgabe als HTML
    chart.save('components_outcome_services.html')

    print("Chart saved as 'components_outcome_services.html'")

    pass
