from pathlib import Path

import pandas as pd
import altair as alt


def crosstab_heatmap(data: pd.DataFrame, col1: str, col2: str, output_path: Path) -> None:
    df = data[[col1, col2]].copy()

    crosstab = pd.crosstab(df['Component'], df['Outcome Description'])
    crosstab['Total'] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values(by='Total', ascending=False).drop(columns='Total')

    # Kreuztabelle in ein DataFrame umwandeln für die Visualisierung
    crosstab_df = crosstab.reset_index().melt(id_vars=f'{col1}', var_name=f'{col2}', value_name='Count')

    # Visualisierung der Kreuztabelle
    heatmap = alt.Chart(crosstab_df).mark_rect().encode(
        alt.X(f'{col1}:O', title=f'{col1}',
              sort=alt.EncodingSortField(field='Count', op='sum', order='descending')),
        alt.Y(f'{col2}:O', title=f'{col2}',
              sort=alt.EncodingSortField(field='Count', op='sum', order='descending')),
        alt.Color('Count:Q',
                  scale=alt.Scale(
                      scheme='inferno',  # Farbskala
                      type='log',  # Logarithmische Skala
                      domain=[1, 10000]  # Wertebereich anpassen
                  ),
                  title='Count'),
        tooltip=[alt.Tooltip(f'{col1}:O', title=f'{col2}'),
                 alt.Tooltip(f'{col2}:O', title=f'{col2}'),
                 alt.Tooltip('Count:Q', title='Count')]
    ).properties(
        title=f'Heatmap of {col1} vs {col2}',
        width=800,
        height=600
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    ).configure_title(
        fontSize=16
    )

    # Speichern der Visualisierung als HTML
    heatmap.save(output_path / f'{col1}_{col2}_heatmap_log.html')
