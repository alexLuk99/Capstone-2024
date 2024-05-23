from pathlib import Path
import pandas as pd

from src.analysis.visuals.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.component_outcome_services import component_outcome_services
from src.analysis.visuals.counts_barchart import counts_barchart, normalized_barchart_log, counts_barchart_log, \
    counts_barchart_color
from src.analysis.visuals.crosstab_heatmap import crosstab_heatmap
from src.analysis.visuals.most_common_word import most_common_word
from src.analysis.visuals.start_end_date import policy_start_end_date
from src.analysis.visuals.timeline import get_timeline


def analyse_data() -> None:
    # CSV-Datei einlesen
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)
    df_assistance = df_assistance.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Difference between policy start/end date with first assistance call
    policy_start_end_date(data=df_assistance, output_path=output_path)

    # Erstellen von Barcharts
    counts_barchart(data=df_assistance, col='Country Of Origin', output_path=output_path)
    counts_barchart(data=df_assistance, col='Outcome Description', output_path=output_path)
    counts_barchart(data=df_assistance, col='Component', output_path=output_path)
    counts_barchart(data=df_assistance, col='Typ aus VIN', output_path=output_path)
    # counts_barchart(data=df_assistance, col='VIN', output_path=output_path)

    # Erstellen von Barcharts mit Farbe
    counts_barchart_color(data=df_assistance, col='Model Year', color='Typ aus VIN', output_path=output_path)
    counts_barchart_color(data=df_assistance, col='Component', color='Outcome Description', output_path=output_path)
    counts_barchart_color(data=df_assistance, col='Model Year', color='Typ aus VIN', output_path=output_path)

    # Extrahiere die gebräuchigsten Wörter aus der Spalter "Fault Description Customer"
    most_common_word(data=df_assistance, output_path=output_path)

    # Barchart mit normalisierte Spalte
    normalized_barchart_log(data=df_assistance, col='VIN', output_path=output_path)

    # Kreuztabelle erstellen und sortieren
    crosstab_heatmap(data=df_assistance, col1='Component', col2='Outcome Description', output_path=output_path)

    # Odometer
    counts_barchart_log(data=df_assistance, col='Odometer', output_path=output_path)

    # Abhängigkeit zwischen Component, Outcome und Services darstellen
    component_outcome_services(data=df_assistance, output_path=output_path)

    # Erstellen von Maps
    create_country_choropleth(df=df_assistance, column='Country Of Origin', title='Number of permits')
    create_country_choropleth(df=df_assistance, column='Country Of Incident', title='Number of incidents')

    # Chi Quadrat tests auf nominalen Attributen
    perform_chi_square_test(data=df_assistance, col1='Component', col2='Outcome Description')
    perform_chi_square_test(data=df_assistance, col1='Component', col2='Reason Of Call')
    perform_chi_square_test(data=df_assistance, col1='Outcome Description', col2='Reason Of Call')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Component')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Outcome Description')
    perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Reason Of Call')

    get_timeline(data=df_assistance, col='Incident Date', aggregate='Monat')


    # Remove rows where 'Reason Of Call' is empty
    df_assistance = df_assistance.dropna(subset=['Reason Of Call'])

    # Count frequency of each category in 'Reason Of Call'
    value_counts = df_assistance['Reason Of Call'].value_counts().reset_index()
    value_counts.columns = ['Reason Of Call', 'Frequency']

    # Create a logarithmic bar chart with Altair
    bar_chart = alt.Chart(value_counts).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency', scale=alt.Scale(type='log')),
        y=alt.Y('Reason Of Call:N', sort='-x', title='Reason Of Call'),
        tooltip=['Reason Of Call', 'Frequency']
    ).properties(
        title='Frequency of Categories in "Reason Of Call"',
        width=800,
        height=400
    )

    # Save the diagram
    bar_chart.save(output_path / 'reason_of_call_frequency_logarithmic.html')

    # Heatmep Bolean
    # List of columns of interest
    columns_of_interest = [
        'Hotel Service', 'Alternative Transport', 'Taxi Service', 'Vehicle Transport',
        'Car Key Service', 'Parts Service', 'Personal Services', 'Damage During Assistance',
        'Additional Services Not Covered'
    ]

    # Ensure all columns are present in the dataframe
    df_selected = df_assistance[columns_of_interest]

    # Count and drop rows with missing values
    initial_row_count = df_selected.shape[0]
    df_selected = df_selected.dropna()
    deleted_row_count = initial_row_count - df_selected.shape[0]
    print(f"Deleted {deleted_row_count} rows with missing values")

    # Convert boolean values to binary
    df_binary = df_selected.applymap(lambda x: 1 if x == 'YES' else 0)

    # Calculate the correlation matrix
    correlation_matrix = df_binary.corr()

    # Melt the correlation matrix for visualization
    correlation_melt = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_melt.columns = ['Variable1', 'Variable2', 'Correlation']

    # Create a heatmap using Altair
    heatmap = alt.Chart(correlation_melt).mark_rect().encode(
        x='Variable1:O',
        y='Variable2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(
        title='Correlation Heatmap of Boolean Variables',
        width=600,
        height=600
    )

    # Save the heatmap
    heatmap.save(output_path / 'correlation_heatmap.html')

    #  replacement cars distribution
    # Ensure the "Replacement Car Days" column is numeric and drop NaN values
    df_assistance['Replacement Car Days'] = pd.to_numeric(df_assistance['Replacement Car Days'], errors='coerce')
    df_assistance = df_assistance.dropna(subset=['Replacement Car Days'])

    # Calculate statistics
    stats = {
        'Mean': df_assistance['Replacement Car Days'].mean(),
        'Median': df_assistance['Replacement Car Days'].median(),
        'Q0.1': df_assistance['Replacement Car Days'].quantile(0.1),
        'Q0.9': df_assistance['Replacement Car Days'].quantile(0.9)
    }

    # Create histogram with hover details and logarithmic scale
    hist = alt.Chart(df_assistance).mark_bar().encode(
        alt.X('Replacement Car Days:Q', bin=alt.Bin(maxbins=20), title='Replacement Car Days'),
        alt.Y('count()', title='Frequency', scale=alt.Scale(type='log')),
        tooltip=[alt.Tooltip('Replacement Car Days:Q', bin=True, title='Replacement Car Days'),
                 alt.Tooltip('count()', title='Frequency')]
    ).properties(
        title='Histogram of Replacement Car Days with Statistical Indicators',
        width=800,
        height=400
    ).interactive()

    # Create rules for mean, median, and quartiles
    stats_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
    stats_df['Color'] = stats_df['Statistic'].apply(
        lambda x: 'red' if x == 'Mean' else 'blue' if x == 'Median' else 'green')

    rules = alt.Chart(stats_df).mark_rule().encode(
        x='Value:Q',
        color=alt.Color('Color:N', scale=None),
        size=alt.value(2),
        tooltip=[alt.Tooltip('Value:Q', title='Value'), alt.Tooltip('Statistic:N', title='Statistic')]
    )

    # Combine the histogram with the statistical indicators
    chart = hist + rules

    # Save the chart
    chart.save(output_path / 'replacement_car_days_distribution.html')


    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
