from pathlib import Path
import pandas as pd
import altair as alt

from src.analysis.statistics.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.timeline import get_timeline


def analyse_data() -> None:
    df_assistance = pd.read_csv('data/interim/assistance.csv')
    df_assistance = df_assistance.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Example 1
    # country_of_orgigin = alt.Chart(df_assistance).mark_bar().encode(
    #     alt.X('Country Of Origin:N', title='Unfallsland'),
    #     alt.Y('count(Country Of Origin):Q', title='Anzahl'),
    #     tooltip=[
    #         alt.Tooltip('Country Of Origin:N', title='Unfallsland'),
    #         alt.Tooltip('count(Country Of Origin):Q', title='Anzahl', format=',.0f'),
    #     ]
    # )
    # country_of_orgigin.save('output/country_of_orgigin_chart.html')
    #
    # # Example 2
    # component = alt.Chart(df_assistance).mark_bar().encode(
    #     alt.X('Component:N', title='Komponente'),
    #     alt.Y('count(Component):Q', title='Anzahl'),
    #     tooltip=[
    #         alt.Tooltip('Component:N', title='Komponente'),
    #         alt.Tooltip('count(Component):Q', title='Anzahl', format=',.0f'),
    #     ]
    # )
    #
    # component.save('output/component_chart.html')
    #
    # # Jan
    # create_country_choropleth(df=df_assistance, column='Country Of Origin', title='Number of permits')
    # create_country_choropleth(df=df_assistance, column='Country Of Incident', title='Number of incidents')

    # perform_chi_square_test(data=df_assistance, col1='Component', col2='Outcome Description')
    # perform_chi_square_test(data=df_assistance, col1='Component', col2='Reason Of Call')
    # perform_chi_square_test(data=df_assistance, col1='Outcome Description', col2='Reason Of Call')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Component')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Outcome Description')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Reason Of Call')

    get_timeline(data=df_assistance, col='Incident Date', aggregate='Monat')


    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
