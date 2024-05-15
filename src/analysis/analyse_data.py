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
    component = alt.Chart(df_assistance['Component']).mark_bar().encode(
        alt.X('Component:N', title='Komponente'),
        alt.Y('count(Component):Q', title='Anzahl'),
        tooltip=[
            alt.Tooltip('Component:N', title='Komponente'),
            alt.Tooltip('count(Component):Q', title='Anzahl', format=',.0f'),
        ]
    )

    component.save('output/component_chart.html')

    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
