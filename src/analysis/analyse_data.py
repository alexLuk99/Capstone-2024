from pathlib import Path
import pandas as pd
import altair as alt

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

    # Merged-Tabelle einlesen
    df_merged = pd.read_csv('data/interim/merged.csv', low_memory=False)
    df_merged = df_merged.convert_dtypes()

    # Create output dir
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Berechne die Anzahl der Werkstattaufenthalte pro VIN
    df_workshop_count = df_merged['VIN'].value_counts().reset_index()
    df_workshop_count.columns = ['VIN', 'Workshop_Count']

    # Merge mit df_merged, um die Anzahl der Werkstattaufenthalte hinzuzufügen
    df_merged = df_merged.merge(df_workshop_count, on='VIN', how='left')

    # Identifiziere die VINs mit hohen Werkstattaufenthalten (z.B. oberes 10% Quantil)
    high_workshop_threshold = df_workshop_count['Workshop_Count'].quantile(0.90)
    high_workshop_vins = df_workshop_count[df_workshop_count['Workshop_Count'] >= high_workshop_threshold]['VIN']

    # Filtern der Einträge mit hohen Werkstattaufenthalten
    df_high_workshop = df_merged[df_merged['VIN'].isin(high_workshop_vins)]

    # Berechne die häufigsten Werkstätten (Händler Q-Line) für diese VINs
    df_high_workshop_count = df_high_workshop['Händler Q-Line'].value_counts().reset_index()
    df_high_workshop_count.columns = ['Händler Q-Line', 'Count']

    # Visualisierung der Verteilung der häufigen Werkstätten
    chart = alt.Chart(df_high_workshop_count).mark_bar().encode(
        x=alt.X('Händler Q-Line:N', title='Händler Q-Line'),
        y=alt.Y('Count:Q', title='Anzahl der Werkstattaufenthalte'),
        tooltip=['Händler Q-Line', 'Count']
    ).properties(
        title='Häufigkeit der Werkstätten (Händler Q-Line) für VINs mit hohen Werkstattaufenthalten',
        width=2600,
        height=2400
    )

    # Speichern der Visualisierung
    chart_path = output_path / 'high_workshop_vins_händler_qline.html'
    chart.save(chart_path)
    print(f"Visualisierung gespeichert unter: {chart_path}")

    # Filter Einträge mit der Q-Line '2600003'
    df_händler = df_merged[df_merged['Händler Q-Line'] == 2600003]

    # Überprüfen, ob Einträge vorhanden sind
    if df_händler.empty:
        print("Keine Einträge mit der Q-Line '2600003' gefunden.")
        return

    # SoS-O-Meter Scores der Autos, die bei diesem Händler waren
    sos_o_meter_scores = df_händler[['VIN', 'SuS-O-Meter']]

    # Ausgabe der SoS-O-Meter Scores
    print("SoS-O-Meter Scores der Autos bei Händler Q-Line 2600003:")
    print(sos_o_meter_scores)

    # Visualisierung der SoS-O-Meter Scores
    chart = alt.Chart(sos_o_meter_scores).mark_bar().encode(
        x=alt.X('VIN:N', title='VIN'),
        y=alt.Y('SuS-O-Meter:Q', title='SoS-O-Meter Score'),
        tooltip=['VIN', 'SuS-O-Meter']
    ).properties(
        title='SoS-O-Meter Scores der Autos bei Händler Q-Line 2600003',
        width=800,
        height=400
    )

    # Speichern der Visualisierung
    chart_path = output_path / 'sos_o_meter_scores_händler_2600003.html'
    chart.save(chart_path)
    print(f"Visualisierung gespeichert unter: {chart_path}")

    # Filter Einträge mit der Q-Line '2600003'
    df_händler = df_merged[df_merged['Händler Q-Line'] == '2600003']

    # Überprüfen, ob Einträge vorhanden sind
    if df_händler.empty:
        print("Keine Einträge mit der Q-Line '2600003' gefunden.")
        return

    # Anzahl der Aufenthalte pro VIN bei der besagten Werkstatt
    vin_counts = df_händler['VIN'].value_counts().reset_index()
    vin_counts.columns = ['VIN', 'Count']

    # SoS-O-Meter Scores der Autos, die bei diesem Händler waren
    sos_o_meter_scores = df_händler[['VIN', 'SuS-O-Meter']].drop_duplicates()

    # Zusammenführen der Daten
    merged_data = pd.merge(vin_counts, sos_o_meter_scores, on='VIN')

    # Visualisierung der SoS-O-Meter Scores als Scatterplot
    scatterplot = alt.Chart(merged_data).mark_point().encode(
        x=alt.X('Count:Q', title='Anzahl der Aufenthalte bei Händler 2600003'),
        y=alt.Y('SuS-O-Meter:Q', title='SoS-O-Meter Score'),
        tooltip=['VIN', 'Count', 'SuS-O-Meter']
    ).properties(
        title='SoS-O-Meter Scores der Autos bei Händler Q-Line 2600003',
        width=800,
        height=400
    )

    # Speichern der Visualisierung
    chart_path = output_path / 'sos_o_meter_scores_händler_2600003_scatter.html'
    scatterplot.save(chart_path)
    print(f"Visualisierung gespeichert unter: {chart_path}")

    # Hinzufügen einer Indikatorspalte, die anzeigt, ob die Q-Line fehlt
    df_merged['Q-Line Missing'] = df_merged['Q-Line'].isna().astype(int)  # Konvertieren zu int

    # Korrelationen zwischen der Indikatorspalte und anderen numerischen Variablen berechnen
    numeric_cols = df_merged.select_dtypes(include=['number']).columns
    correlation_matrix = df_merged[numeric_cols].corr()

    # Korrelationen mit 'Q-Line Missing' extrahieren
    q_line_corr = correlation_matrix['Q-Line Missing'].drop('Q-Line Missing').sort_values(ascending=False)

    # Ausgabe der Korrelationen
    print(q_line_corr)

    # Umwandlung der Korrelationsmatrix in ein langes Format
    corr_matrix_long = correlation_matrix.reset_index().melt(id_vars='index')
    corr_matrix_long.columns = ['Variable1', 'Variable2', 'Correlation']

    # Visualisierung der Korrelationsmatrix mit Altair
    base = alt.Chart(corr_matrix_long).encode(
        x=alt.X('Variable1:O', title='Variable 1'),
        y=alt.Y('Variable2:O', title='Variable 2')
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    )

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlation > 0.5,  # Helle Farbe für positive Korrelationen
            alt.value('black'),
            alt.value('white')
        )
    )

    chart = heatmap + text

    chart = chart.properties(
        title='Correlation Heatmap',
        width=600,
        height=600
    )

    # Speichern der Heatmap
    heatmap_path = output_path / 'correlation_heatmapppppp.html'
    chart.save(heatmap_path)
    print("Heatmap saved as", heatmap_path)

    # Filter nur numerische Spalten für die Berechnung der Korrelationsmatrix
    numeric_cols = df_merged.select_dtypes(include=['number']).columns
    df_numeric = df_merged[numeric_cols]

    # Berechnung der Korrelationsmatrix
    corr_matrix = df_numeric.corr()

    # Umwandlung der Korrelationsmatrix in ein langes Format
    corr_matrix_long = corr_matrix.reset_index().melt(id_vars='index')
    corr_matrix_long.columns = ['Variable1', 'Variable2', 'Correlation']

    # Visualisierung der Korrelationsmatrix mit Altair
    base = alt.Chart(corr_matrix_long).encode(
        x=alt.X('Variable1:O', title='Variable 1'),
        y=alt.Y('Variable2:O', title='Variable 2')
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    )

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlation > 0.5,  # Helle Farbe für positive Korrelationen
            alt.value('black'),
            alt.value('white')
        )
    )

    chart = heatmap + text

    chart = chart.properties(
        title='Correlation Heatmap',
        width=600,
        height=600
    )

    # Speichern der Heatmap
    heatmap_path = output_path / 'correlation_heatmap.html'
    chart.save(heatmap_path)
    print("test")

    # SuS-Spalten auswählen
    sus_columns = [
        'SuS_Abschleppungen', 'SuS_Anruferzahl', 'SuS_Vertragszeitraum',
        'SuS_Services_Offered', 'SuS_AnrufeInFall', 'SuS_Rental_Car',
        'SuS_Breakdown', 'SuS-O-Meter'
    ]

    # Filter nur die SuS-Spalten
    df_sus = df_assistance[sus_columns]

    # Berechnung der Korrelationsmatrix
    corr_matrix = df_sus.corr()

    # Umwandlung der Korrelationsmatrix in ein langes Format
    corr_matrix_long = corr_matrix.reset_index().melt(id_vars='index')
    corr_matrix_long.columns = ['Variable1', 'Variable2', 'Correlation']

    # Visualisierung der Korrelationsmatrix mit Altair
    base = alt.Chart(corr_matrix_long).encode(
        x=alt.X('Variable1:O', title='Variable 1'),
        y=alt.Y('Variable2:O', title='Variable 2')
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    )

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlation > 0.5,  # Helle Farbe für positive Korrelationen
            alt.value('black'),
            alt.value('white')
        )
    )

    chart = heatmap + text

    chart = chart.properties(
        title='Correlation Heatmap of SuS Columns',
        width=600,
        height=600
    )

    # Speichern der Heatmap
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)
    chart.save(output_path / 'sus_correlation_heatmap.html')
    print("test")

    # Top 10% der VINs mit den meisten Abschleppvorgängen
    towing_df = df_assistance[df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])]
    vin_counts = towing_df['VIN'].value_counts()
    threshold = vin_counts.quantile(0.90)
    top_10_percent_vins = vin_counts[vin_counts >= threshold]
    top_10_percent_vins_df = top_10_percent_vins.reset_index()
    top_10_percent_vins_df.columns = ['VIN', 'Count']

    # Sortieren nach der Anzahl der Abschleppvorgänge
    top_10_percent_vins_df = top_10_percent_vins_df.sort_values(by='Count', ascending=False).head(100)

    # Erstellen des Balkendiagramms
    chart = alt.Chart(top_10_percent_vins_df).mark_bar().encode(
        x=alt.X('VIN:N', sort='-y', title='VIN'),
        y=alt.Y('Count:Q', title='Anzahl der Abschleppvorgänge'),
        tooltip=['VIN:N', 'Count:Q']
    ).properties(
        title='Top 10% der VINs mit den meisten Abschleppvorgängen',
        width=800,
        height=400
    )

    # Speichern des Balkendiagramms
    chart.save(output_path / 'top_10_percent_vins_towing.html')
    print("test")

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
