from typing import Literal

import pandas as pd
import altair as alt


def get_timeline(data: pd.DataFrame, col: str, aggregate: Literal['Monat', 'Quartal', 'Quartal_Monat']) -> None:
    """
    Get a timeline of data
    :param data: The input DataFrame containing the Date data.
    :param col: The name of the date column.
    :param aggregate: Literal one of 'Monat', 'Quartal', 'Quartal_Monat'. How to aggregate the data.
    :return: None (alt.Chart)
    """
    df = pd.DataFrame(data[col]).copy()

    aggregate_dict = {'Monat': 'yearmonth',
                      'Quartal': 'yearquarter',
                      'Quartal_Monat': 'yearquartermonth'}

    agg = aggregate_dict.get(aggregate)

    chart = alt.Chart(df).mark_line(
        point=alt.OverlayMarkDef(filled=False, fill="white")
    ).encode(
        x=alt.X(f'{agg}({col}):T', title=f'{col}'),
        y=alt.Y('count():Q', title='Incident Count'),
        tooltip=[
            alt.Tooltip(f'{agg}({col}):T', title='Datum'),
            alt.Tooltip('count():Q', title='Anzahl'),
        ]
    ).properties(
        title=f'{col} Count Over Time'
    )

    chart.save(f'output/{col}_Count_{agg}.html')
