from pathlib import Path
import pandas as pd
import altair as alt
import geopandas as gpd


def analyse_data() -> None:
    df_assistance = pd.read_csv('data/interim/assistance.csv')
    df_assistance = df_assistance.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Load GeoJSON data for Europe
    url = 'https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson'
    geojson = gpd.read_file(url)

    geo_df = pd.DataFrame(geojson)
    count_country_of_origin = \
    df_assistance[['Country Of Origin', 'VIN']].groupby(by='Country Of Origin', as_index=False)['VIN'].count()
    geo_df = geo_df.merge(count_country_of_origin, left_on='ISO2', right_on='Country Of Origin', how='left')
    geo_df['VIN'] = geo_df['VIN'].fillna(0)

    map = alt.Chart(alt.Data(values=geo_df.to_dict(orient='records'))).mark_geoshape(
        color='Country Of Origin:Q'
    ).transform_lookup(
        lookup='ISO2',
        from_=alt.LookupData(count_country_of_origin, 'Country Of Origin', list(count_country_of_origin.columns))
    ).project(
        type='mercator'
    )

    map.save('output/map.html')

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
