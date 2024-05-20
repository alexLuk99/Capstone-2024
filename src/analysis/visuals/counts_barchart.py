from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm


def counts_barchart(data: pd.DataFrame, col: str, output_path: Path) -> None:
    df = pd.DataFrame(data[col]).copy()

    counts = df[f'{col}'].value_counts().reset_index()
    counts.columns = [f'{col}', 'Frequency']

    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency'),
        y=alt.Y(f'{col}:N', sort='-x', title=f'{col}'),
        tooltip=[f'{col}', 'Frequency']
    ).properties(
        title=f'Frequency of {col}',
        width=800,
        height=400
    )

    chart.save(output_path / f'{col}_count.html')


def counts_barchart_log(data: pd.DataFrame, col: str, output_path: Path) -> None:
    df = pd.DataFrame(data[col]).copy()

    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f'{col}:Q', bin=alt.Bin(maxbins=40), title=f'{col}'),
        alt.Y('count()', title='Anzahl der Fahrzeuge', scale=alt.Scale(type='log')),
        tooltip=[alt.Tooltip(f'{col}:Q', title='Kilometerstand', bin=alt.Bin(maxbins=40)),
                 alt.Tooltip('count()', title='Anzahl der Fahrzeuge')]
    ).properties(
        title=f'Verteilung der Spalte {col} (logarithmische Skala)'
    )

    chart.save(output_path / f'{col}_count_log.html')


def normalized_barchart_log(data: pd.DataFrame, col: str, output_path: Path) -> alt.Chart:
    df = pd.DataFrame(data[col]).copy()

    counts = df[f'{col}'].value_counts().reset_index()
    counts.columns = [f'{col}', 'Frequency']

    hist = alt.Chart(counts).mark_bar().encode(
        alt.X('Frequency:Q', bin=alt.Bin(maxbins=30), title='Frequency'),
        alt.Y('count():Q', scale=alt.Scale(type='log'), title='Count (log scale)'),
        tooltip=[alt.Tooltip('Frequency:Q', title='Frequency'), alt.Tooltip('count():Q', title='Count')]
    ).properties(
        title=f'Histogram of {col} Frequencies (mit Log Scale)',
        width=800,
        height=400
    )

    mean = counts['Frequency'].mean()
    std_dev = counts['Frequency'].std()
    x = np.linspace(counts['Frequency'].min(), counts['Frequency'].max(), 100)
    y = norm.pdf(x, mean, std_dev)
    normal_dist_df = pd.DataFrame({'Frequency': x, 'Density': y})

    line = alt.Chart(normal_dist_df).mark_line(color='red').encode(
        x='Frequency:Q',
        y=alt.Y('Density:Q'),
        tooltip=[alt.Tooltip('Frequency:Q', title='Frequency'), alt.Tooltip('Density:Q', title='Density')]
    )

    # Kombiniertes Diagramm
    combined_chart = alt.layer(hist, line).resolve_scale(
        y='independent'
    ).properties(
        title=f'{col} Frequency Distribution with Normal Curve (Log Scale)'
    )

    # Speichern der Visualisierung als HTML
    combined_chart.save(output_path / f'{col}_frequency_normal_distribution_log.html')

    # Ausgabe der Häufigkeiten als CSV
    counts.to_csv(output_path / f'{col}_frequency_counts.csv', index=False)

    # print(counts['Frequency'].describe())


def counts_barchart_color(data: pd.DataFrame, col: str, color: str, output_path: Path) -> None:
    df = data[[col, color]].copy()

    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f'{col}:N', title=f'{col}'),
        alt.Y(f'count({col}):Q', title='Anzahl'),
        color=f'{color}:N',
        tooltip=[
            alt.Tooltip(f'{color}:N', title=f'{color}'),
            alt.Tooltip(f'{col}:N', title=f'{col}'),
            alt.Tooltip(f'count({col}):Q', title='Anzahl', format=',.0f'),
        ]
    )

    chart.save(output_path / f'{col}_{color}_chart.html')
