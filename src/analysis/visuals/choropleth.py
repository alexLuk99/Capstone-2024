from pathlib import Path

import pandas as pd
import altair as alt
import requests


def create_country_choropleth(df: pd.DataFrame, column: str, output_path: Path) -> None:
    # Load GeoJSON data for Europe
    url = 'https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson'
    response = requests.get(url)
    europe_geojson = response.json()

    # Prepare Data
    country_counts = df[column].value_counts().reset_index()

    # Count the number of country occurences
    country_counts.columns = ['Country', 'Count']

    country_codes = pd.read_csv('utils/mapping/country_codes.csv')
    country_codes = dict(zip(country_codes['Alpha-2 Code'], country_codes['Country Name']))
    # Add full country names to the country_counts DataFrame
    country_counts['Country Name'] = country_counts['Country'].map(country_codes, na_action='ignore')

    choropleth = alt.Chart(alt.Data(values=europe_geojson['features'])).mark_geoshape(stroke='white').encode(
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='tealblues', type='log'),
                        legend=alt.Legend(title='Number of VINs')),
        tooltip=[
            alt.Tooltip('properties.NAME:N', title='Country'),
            alt.Tooltip('Count:Q', title='Number of IDs')
        ]
    ).transform_lookup(
        lookup='properties.NAME',
        from_=alt.LookupData(country_counts, 'Country Name', ['Count'])
    ).properties(
        width=800,
        height=600,
        title='Choropleth Map of European Countries by Number of VINs'
    ).project(
        type='mercator'
    )

    maps_path = output_path / 'maps'
    maps_path.mkdir(exist_ok=True, parents=True)

    choropleth.save(maps_path / f'europe_choropleth_{column}.html')
