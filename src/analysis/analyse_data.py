from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime, timedelta

def analyse_data() -> None:
    # Option 1: Load your existing data
    df_assistance = pd.read_csv('data/interim/assistance.csv')
    df_assistance = df_assistance.convert_dtypes()
    print(df_assistance)

    # Convert date columns, handling errors and filtering out invalid dates
    for col in ['Incident Date', 'Policy End Date']:
        if col == 'Incident Date':
            format = '%Y-%m-%d %H:%M:%S'  # Format for 'Incident Date' (YYYY-MM-DD HH:MM:SS)
        elif col == 'Policy End Date':
            format = '%Y-%m-%d'  # Format for 'Policy End Date' (YYYY-MM-DD)

        df_assistance[col] = pd.to_datetime(df_assistance[col], format=format, errors='coerce')
        df_assistance.dropna(subset=[col], inplace=True)

    print(df_assistance)

    # Calculate difference in days, filtering out negative/zero values (same for both options)
    df_assistance['Days Until Policy End'] = (df_assistance['Policy End Date'] - df_assistance['Incident Date']).dt.days
    df_assistance = df_assistance[df_assistance['Days Until Policy End'] > 0]

    # Display the first 5 rows after processing to check (optional)
    print(df_assistance[['Incident Date', 'Policy End Date', 'Days Until Policy End']].head().to_markdown(index=False, numalign="left", stralign="left"))

    # Create Altair chart for incidents relative to policy end date (without binning)
    chart = alt.Chart(df_assistance).mark_bar().encode(
        x=alt.X('Days Until Policy End:Q', title='Days Until Policy End'),
        y=alt.Y('count()', title='Number of Incidents'),
        tooltip=['Days Until Policy End', 'count()']
    ).properties(
        title='Distribution of Incidents Relative to Policy End Date (No Bins)',
        width=800,  # Breite in Pixel
        height=400   # Höhe in Pixel
    ).interactive()


    chart.save('output/incidents_vs_policy_end_histogram.html')

    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
