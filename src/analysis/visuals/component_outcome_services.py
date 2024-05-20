from pathlib import Path

import altair as alt
import pandas as pd


def component_outcome_services(data: pd.DataFrame, output_path: Path) -> alt.Chart:
    df = data.copy()

    melted_df = df.melt(id_vars=['Component', 'Outcome Description'],
                        value_vars=['Hotel Service', 'Alternative Transport', 'Taxi Service',
                                    'Vehicle Transport', 'Car Key Service', 'Parts Service'],
                        var_name='Service', value_name='Angeboten')

    # Häufigkeiten berechnen
    freq_df = melted_df.groupby(['Component', 'Outcome Description', 'Service', 'Angeboten']).size().reset_index(
        name='Count')

    freq_df = freq_df.convert_dtypes()

    selection = alt.selection_point(fields=['Angeboten'], bind='legend')

    # Erstellen eines gestapelten Balkendiagramms
    chart = alt.Chart(freq_df).mark_bar().encode(
        x='Component:N',
        y='Count:Q',
        color=alt.Color('Angeboten:N', scale=alt.Scale(domain=['NO', 'YES'])),
        tooltip=['Component:N', 'Outcome Description:N', 'Service:N', 'Angeoten:N', 'Count:Q']
    ).facet(
        column=alt.Column('Service:N', header=alt.Header(labelAngle=-90, title='Service'))
    ).properties(
        title='Influence of Components and Outcome on Services'
    ).add_params(
        selection
    ).transform_filter(
        selection
    )

    # Ausgabe als HTML
    chart.save(output_path / 'components_outcome_services.html')
