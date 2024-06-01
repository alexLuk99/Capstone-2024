import re
import numpy as np
from loguru import logger

import pandas as pd
import altair as alt
from scipy.stats import stats

from config.paths import input_path, interim_path, output_path


def format_time(time_str):
    if pd.isna(time_str):
        return time_str
    match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if match:
        return match.group(1).zfill(2) + ':' + match.group(2) + ':00'
    return pd.NA


def read_prepare_data() -> pd.DataFrame:
    # Read and prepare assistance file
    # column Monat and License Plate are missing in sheet 2023
    # some other columns are named differently in different sheets -> consistent/uniform naming
    # column a and b don't contain any values and only exist in sheet 2023 -> drop them
    logger.info('Prepare assistance report ... ')
    sheets = ['2021', '2022', '2023']
    assistance_list = []
    for sheet in sheets:
        tmp = pd.read_excel(open(input_path / 'Assistance_Report_Europa_2021-2023_anonymized.xlsx', 'rb'),
                            sheet_name=sheet)
        if 'Product  Type' in tmp.columns:
            tmp = tmp.rename(columns={'Product  Type': 'Report Type'})
        if 'a' in tmp.columns:
            tmp = tmp.drop(columns=['a'])
        if 'b' in tmp.columns:
            tmp = tmp.drop(columns=['b'])
        if 'CountryOfOrigin' in tmp.columns:
            tmp = tmp.rename(columns={'CountryOfOrigin': 'Country Of Origin'})
        if 'RSACarried Out By' in tmp.columns:
            tmp = tmp.rename(columns={'RSACarried Out By': 'RSA Carried Out By'})
        if 'RSASuccessful' in tmp.columns:
            tmp = tmp.rename(columns={'RSASuccessful': 'RSA Successful'})
        if 'Reason Courtesy Call ' in tmp.columns:
            tmp = tmp.rename(columns={'Reason Courtesy Call ': 'Reason Courtesy Call'})
        assistance_list.append(tmp)
    # concat the three excel sheets to singel dataframe
    df_assistance = pd.concat(assistance_list, ignore_index=True)
    # Convert dtypes to pandas dtypes
    df_assistance = df_assistance.convert_dtypes()

    # Drop duplicates over whole dataset (however, there are duplicate case numbers)
    df_assistance = df_assistance.drop_duplicates()

    # Convert Incident Date and Time of Arrival to DateTime and add time from Time of Call or Time of Arrival
    df_assistance['Incident Date'] = pd.to_datetime(df_assistance['Incident Date'], format='%d/%m/%Y')
    df_assistance['Time Of Arrival'] = df_assistance['Time Of Arrival'].replace({'0': pd.NA})
    df_assistance['Time Of Arrival'] = df_assistance['Time Of Arrival'].replace({'00:00': pd.NA})
    df_assistance['Time Of Arrival'] = df_assistance['Time Of Arrival'].apply(format_time)

    df_assistance.loc[~df_assistance['Time Of Arrival'].isna(), 'Time Of Arrival'] = df_assistance.loc[~df_assistance[
        'Time Of Arrival'].isna(), 'Incident Date'] + pd.to_timedelta(
        df_assistance.loc[~df_assistance['Time Of Arrival'].isna(), 'Time Of Arrival'])
    df_assistance['Time Of Arrival'] = pd.to_datetime(df_assistance['Time Of Arrival'])
    df_assistance['Incident Date'] = df_assistance['Incident Date'] + pd.to_timedelta(
        df_assistance['Time Of Call'] + ':00')

    # Drop duplicates over VIN and Incident Date
    df_assistance = df_assistance.drop_duplicates(subset=['VIN', 'Incident Date'])

    # replace CTA with our calculated difference between Time Of Arrival and Incident Date
    df_assistance['CTA'] = df_assistance['Time Of Arrival'] - df_assistance['Incident Date']
    df_assistance.loc[df_assistance['CTA'] <= pd.Timedelta(0), 'CTA'] = pd.NaT
    df_assistance['CTA'] = (df_assistance['CTA'].dt.total_seconds() // 60).fillna(0).astype('Int64').replace(0, pd.NA)

    date_columns = ['Registration Date', 'Policy Start Date', 'Policy End Date']
    for column in date_columns:
        # one record with typo and value of 28/05/2921 in Registration Date
        registration_date_1 = pd.to_datetime(df_assistance[column], errors='coerce', format='%d/%m/%Y')
        registration_date_2 = pd.to_datetime(df_assistance[column], errors='coerce', format='%Y%m%d')
        df_assistance[column] = registration_date_1.fillna(registration_date_2)

    # Add month in month column where value is na
    # Column month does not exist for sheet 2023 (thus some of the missing values)
    df_assistance.loc[df_assistance['Monat'].isna(), 'Monat'] = df_assistance['Incident Date'].dt.month

    # Map Reason Of Call
    # create new column Reson of Call Mapped for mapping
    # fill new column na values with values from column Reason Of Call
    # drop Reason Of Call and rename Reason of Call Mapped in Reason Of Call
    mapping_roc = {
        '1': 'Breakdown - Technical defect',
        '2': 'Breakdown - Misuse',
        '3': 'Accident',
        '4': 'Others',
        '5': 'Theft',
        '6': 'Vandalism',
        '7': 'Tyre Breakdown',
        '8': 'Tyre Accident'
    }

    df_assistance['Reason Of Call Mapped'] = df_assistance['Reason Of Call'].map(mapping_roc)
    df_assistance.loc[df_assistance['Reason Of Call Mapped'].isna(), 'Reason Of Call Mapped'] = df_assistance[
        'Reason Of Call']
    df_assistance = df_assistance.drop(columns=['Reason Of Call'])
    df_assistance = df_assistance.rename(columns={'Reason Of Call Mapped': 'Reason Of Call'})

    # Odometer cleaning
    # Q&A 3 -> max plausible Odometer is 260_000
    df_assistance.loc[df_assistance['Odometer'] < 50, 'Odometer'] = np.nan
    df_assistance.loc[df_assistance['Odometer'] > 260_000, 'Odometer'] = np.nan

    # Mapping vehicle model & Merge
    mapping_vehicle_model = pd.read_excel('utils/mapping/vehicle_model.xlsx')
    mapping_vehicle_model = mapping_vehicle_model.drop(columns=['Bestelltyp'])
    mapping_vehicle_model = mapping_vehicle_model.drop_duplicates()  # No good mapping from Porsche they have duplicates
    mapping_vehicle_model = mapping_vehicle_model.astype(str)
    btypes = mapping_vehicle_model['Bestelltypschlüssel'].unique().tolist()

    df_assistance.loc[~df_assistance['Btype'].isin(btypes), 'Btype'] = pd.NA

    df_assistance_btype_na = df_assistance[df_assistance['Btype'].isna()].convert_dtypes()
    df_assistance_btype_not_na = df_assistance[~df_assistance['Btype'].isna()].convert_dtypes()

    # Merge on Btype
    df_assistance_merge_on_btype = df_assistance_btype_not_na.merge(mapping_vehicle_model, left_on='Btype',
                                                                    right_on='Bestelltypschlüssel', how='left')

    # Merge on VIN
    mapping_vehicle_model_typ_aus_VIN = mapping_vehicle_model.copy()
    mapping_vehicle_model_typ_aus_VIN[['Bestelltypschlüssel', 'Fahrzeuggruppe']] = pd.NA
    mapping_vehicle_model_typ_aus_VIN = mapping_vehicle_model_typ_aus_VIN.drop_duplicates()
    mapping_vehicle_model_typ_aus_VIN = mapping_vehicle_model_typ_aus_VIN.astype(str)
    df_assistance_merge_on_typ_aus_VIN = df_assistance_btype_na.merge(mapping_vehicle_model_typ_aus_VIN,
                                                                      left_on='Typ aus VIN', right_on='Baureihe',
                                                                      how='left')

    df_assistance = pd.concat([df_assistance_merge_on_btype, df_assistance_merge_on_typ_aus_VIN], ignore_index=True)
    df_assistance[['Bestelltypschlüssel', 'Modellreihe', 'Baureihe', 'Fahrzeuggruppe']] = df_assistance[
        ['Bestelltypschlüssel', 'Modellreihe', 'Baureihe', 'Fahrzeuggruppe']].replace({np.nan: pd.NA})

    # Rename
    df_assistance = df_assistance.rename(columns={'Replacement Car Days': 'Rental Car Days'})

    # Policy Duration
    # ToDo: Alle Daten mit Policy Start Date vor Gründung von Porsche Assistance mit pd.NaT ersetzen
    # Porsche Assistance has a maximum duration of 3 years, so all Policy End Dates which are greater than 01.01.2027 are unrealistic
    df_assistance.loc[df_assistance['Policy Start Date'] < pd.to_datetime('2002-01-01',
                                                                          format='%Y-%m-%d'), 'Policy Start Date'] = pd.NaT
    df_assistance.loc[
        df_assistance['Policy End Date'] >= pd.to_datetime('2027-01-01', format='%Y-%m-%d'), 'Policy End Date'] = pd.NaT
    df_assistance.loc[
        df_assistance['Policy End Date'] <= df_assistance['Policy Start Date'], 'Policy End Date'] = pd.NaT

    df_assistance['Policy Duration'] = df_assistance['Policy End Date'] - df_assistance['Policy Start Date']
    df_assistance.loc[df_assistance['Policy Duration'] <= pd.Timedelta(0), 'Policy Duration'] = pd.NaT

    # drop Typ aus VIN, Btype, Vehicle Model as they are now duplicated
    # drop other columns like free text columns, columns no longer used (Personal Services)
    df_assistance = df_assistance.drop(columns=['Typ aus VIN', 'Btype', 'Vehicle Model', 'Monat',
                                                'Fault Description Customer', 'License Plate', 'Model Year',
                                                'Towing Distance', 'Repairing Dealer Code', 'Replacement Car Brand',
                                                'Replacement Car Type', 'Replacement Car Delivered By',
                                                'Personal Services', 'Additional Services Not Covered Description',
                                                'Result Courtesy Call', 'Reason Courtesy Call',
                                                'Time Of Completion Minutes', 'Time Of Call'])

    sort = ['Case Number', 'VIN', 'Report Type', 'Incident Date', 'Registration Date', 'Policy Start Date',
            'Policy End Date', 'Policy Duration', 'Country Of Origin', 'Country Of Incident', 'Handling Call Center',
            'Bestelltypschlüssel', 'Baureihe', 'Modellreihe', 'Fahrzeuggruppe', 'Odometer', 'Time Of Arrival', 'CTA',
            'Component', 'Outcome Description', 'Reason Of Call']

    _ = [sort.append(x) for x in df_assistance.columns if x not in sort]

    df_assistance = df_assistance[sort]

    # The following lines add a new column "Fall_ID" to the df_assistance dataset which can be used as "unique"
    # Identifier to find pairs/groups of calls which happened in a short timespan of each other (6 days) and be regarded
    # as belonging to the same problem with car. Only if multiple calls in the span of 6 days have the outcome
    # description "Towing" or "Scheduled Towing", they receive a new "Fall_ID". After every towing a car should be in a
    # workshop (thus every "Fall_ID" with the Outcome Description "Towing" or "Scheduled Towing" should have an entry
    # (or multiple entries) in the workshop file). Following this, only Fall_IDs with "Towing" or "Scheduled Towing" for
    # their last entry can be merged with df_workshop

    # Sortieren des DataFrames
    df_assistance = df_assistance.sort_values(by=['VIN', 'Incident Date'])

    # Berechnung des Zeitunterschieds
    df_assistance['Time_Diff'] = df_assistance.groupby('VIN')['Incident Date'].diff().dt.days

    # Identifizierung neuer Fälle
    df_assistance['New_Fall'] = (df_assistance['Time_Diff'] > 6) | (df_assistance['Time_Diff'].isna())

    # Bedingung für "Towing" oder "Scheduled Towing"
    towing_condition = df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])

    # Markieren des nächsten Eintrags nach "Towing" oder "Scheduled Towing" als neuer Fall
    df_assistance.loc[towing_condition.shift(fill_value=False), 'New_Fall'] = True

    # Kumulative Summe der neuen Fälle
    df_assistance['Fall_Number'] = df_assistance.groupby('VIN')['New_Fall'].cumsum()

    # Erstellen der Fall_ID
    df_assistance['Fall_ID'] = df_assistance['VIN'] + '_' + df_assistance['Fall_Number'].astype(str)

    # Entfernen temporärer Spalten
    df_assistance = df_assistance.drop(columns=['Time_Diff', 'New_Fall', 'Fall_Number'])

    # Implementierung einer neuen Spalte in Assistance_df für Kennzeichnung der Top x-Prozent, mit boolenschen Wert

    # Obere Prozentzahl
    x = 20

    # Zählen der Häufigkeit jedes eindeutigen Wertes in der Spalte "VIN"
    vin_counts = df_assistance['VIN'].value_counts()

    # Identifizieren der Schwelle für die obersten x%
    threshold = vin_counts.quantile((100 - x) / 100)

    # Erstellen einer Liste der VINs, die in die obersten x% fallen
    top_percent_vins = vin_counts[vin_counts >= threshold].index

    # Erstellen der neuen Spalte "SuS_Anruferzahl" mit dem booleschen Wert "yes" für die obersten x%
    df_assistance['SuS_Anruferzahl'] = df_assistance['VIN'].apply(
        lambda vin: True if vin in top_percent_vins else False)

    # Implementierung einer neuen Spalte in Assistance_df für Kennzeichnung ob Incident Date an den Rändern des Policy Start und End Dates liegt, mit boolenschen Wert
    # Definieren der Bedingungen

    condition_start_date = (df_assistance['Incident Date'] >= df_assistance['Policy Start Date']) & \
                           (df_assistance['Incident Date'] <= df_assistance['Policy Start Date'] + pd.Timedelta(
                               days=30))

    condition_end_date = (df_assistance['Incident Date'] >= df_assistance['Policy End Date'] - pd.Timedelta(days=30)) | \
                         (df_assistance['Incident Date'] > df_assistance['Policy End Date'])

    # Erstellen der neuen Spalte "SuS_Vertragszeitraum"
    df_assistance['SuS_Vertragszeitraum'] = (condition_start_date | condition_end_date)

    # Check for top 10% VINs with most services offered
    offered_services_cols = ['Rental Car', 'Hotel Service', 'Alternative Transport', 'Taxi Service',
                             'Vehicle Transport', 'Car Key Service', 'Parts Service', 'Additional Services Not Covered']

    # Reformat cols in original dataframe
    for col in offered_services_cols:
        df_assistance[col] = df_assistance[col].fillna('NO')
        df_assistance[col] = df_assistance[col].str.upper().astype(str)
        df_assistance[col] = df_assistance[col].map({'YES': True, 'NO': False}).astype(bool)
        df_assistance[col] = df_assistance[col].apply(lambda x: x if isinstance(x, bool) else False)

    # Create temporary dataframe for calculation
    cols_to_keep = offered_services_cols.copy()
    cols_to_keep.append('VIN')
    tmp_assistance = df_assistance[cols_to_keep].copy()

    tmp_assistance_grouped = tmp_assistance.groupby(by='VIN', as_index=False).agg(
        Rental_Car=pd.NamedAgg(column='Rental Car', aggfunc='sum'),
        Hotel_Service=pd.NamedAgg(column='Hotel Service', aggfunc='sum'),
        Alternative_Transport=pd.NamedAgg(column='Alternative Transport', aggfunc='sum'),
        Taxi_Service=pd.NamedAgg(column='Taxi Service', aggfunc='sum'),
        Vehicle_Transport=pd.NamedAgg(column='Vehicle Transport', aggfunc='sum'),
        Car_Key_Service=pd.NamedAgg(column='Car Key Service', aggfunc='sum'),
        Parts_Service=pd.NamedAgg(column='Parts Service', aggfunc='sum'),
        Additional_Services_Not_Covered=pd.NamedAgg(column='Additional Services Not Covered', aggfunc='sum')
    )

    cols = ['Rental_Car', 'Hotel_Service', 'Alternative_Transport', 'Taxi_Service', 'Vehicle_Transport',
            'Car_Key_Service', 'Parts_Service', 'Additional_Services_Not_Covered']

    tmp_assistance_grouped['Total Services Offered'] = tmp_assistance_grouped[cols].sum(axis=1)
    top_percent_services_offered = tmp_assistance_grouped['Total Services Offered'].quantile(0.9)
    tmp_assistance_grouped['SuS_Services_Offered'] = tmp_assistance_grouped[
                                                         'Total Services Offered'] >= top_percent_services_offered

    # Merge SuS_Services_Offered back to original df_assistance dataframe
    df_assistance = df_assistance.merge(tmp_assistance_grouped[['VIN', 'SuS_Services_Offered']], on='VIN', how='left')

    # Welche VIN bekommt besonders oft einen Ersatzwagen
    # Implementierung einer neuen Spalte in Assistance_df für Kennzeichnung der obersten x Prozent, die am meisten Voranrufe in einem Fall haben, mit booleschen Wert

    # Quantile
    quant = 0.75

    # Berechnen der Anzahl der Vorkommen jeder 'Fall_ID'
    fall_id_counts = df_assistance['Fall_ID'].value_counts()

    # Berechnen des Schwellenwertes für die obersten 25%
    threshold = fall_id_counts.quantile(quant)

    # Erstellen der neuen Spalte 'SuS_AnrufeInFall'
    df_assistance['SuS_AnrufeInFall'] = df_assistance['Fall_ID'].map(fall_id_counts) > threshold

    # Welche VIN bekommt besonders oft einen Ersatzwagen
    rental_counts = df_assistance[df_assistance['Rental Car Days'] > 0].groupby('VIN').size()

    # Schwellenwert für die oberen 25 % bestimmen
    threshold = rental_counts.quantile(quant)

    # Boolean-Spalte erstellen, die angibt, ob eine VIN zu den oberen 30 % gehört
    df_assistance['SuS_Rental_Car'] = df_assistance['VIN'].apply(lambda x: rental_counts.get(x, 0) > threshold)

    # Berechne die Top 25% der VINs nach Anzahl der Abschleppvorgänge
    towing_df = df_assistance[df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])]
    vin_counts = towing_df['VIN'].value_counts()
    threshold = vin_counts.quantile(quant)
    top_vins_towing = vin_counts[vin_counts >= threshold].index

    # Berechne, ob dasselbe Auto innerhalb von 14 Tagen nach einem Towing oder Scheduled Towing erneut abgeschleppt wurde
    df_assistance = df_assistance.sort_values(by=['VIN', 'Incident Date'])

    # Markiere Towing und Scheduled Towing
    df_assistance['is_towing'] = df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])

    # Berechne die Differenz in Tagen zwischen aufeinanderfolgenden Towing-Events pro VIN
    df_assistance['days_since_last_towing'] = df_assistance.groupby('VIN')['Incident Date'].diff().dt.days

    # Markiere die Einträge als True, wenn die Differenz 14 Tage oder weniger beträgt
    df_assistance['SuS_Breakdown'] = df_assistance['days_since_last_towing'].le(14) & df_assistance['is_towing']

    # Fülle NaN-Werte in SuS_Breakdown mit False
    df_assistance['SuS_Breakdown'] = df_assistance['SuS_Breakdown'].fillna(False)

    # Entferne die Hilfsspalten
    df_assistance.drop(columns=['is_towing', 'days_since_last_towing'], inplace=True)

    # Füge die neue Spalte SuS_Abschleppungen hinzu
    df_assistance['SuS_Abschleppungen'] = df_assistance['VIN'].apply(lambda vin: vin in top_vins_towing)

    sus_columns = ['SuS_Anruferzahl', 'SuS_Vertragszeitraum', 'SuS_Services_Offered', 'SuS_AnrufeInFall',
                   'SuS_Rental_Car', 'SuS_Breakdown', 'SuS_Abschleppungen']

    df_assistance['SuS-O-Meter'] = df_assistance[sus_columns].sum(axis=1) / len(sus_columns)

    df = pd.DataFrame(df_assistance['SuS-O-Meter'])
    mean = df['SuS-O-Meter'].mean()
    std = df['SuS-O-Meter'].std()
    df['SuS-O-Meter-Standardized'] = (df['SuS-O-Meter'] - mean) / std

    chart = alt.Chart(df).mark_bar().encode(
        alt.X('SuS-O-Meter:Q', bin=True, title='SuS-O-Meter'),
        alt.Y('count()', title='Anzahl')
    ).properties(
        title='Verteilung des SuS-O-Meters'
    )

    chart_stand = alt.Chart(df).mark_bar().encode(
        alt.X('SuS-O-Meter-Standardized:Q', bin=True, title='Normalisierte SuS-O-Meter'),
        alt.Y('count()', title='Anzahl')
    ).properties(
        title='Verteilung des standardisierten SuS-O-Meters'
    )

    box = alt.Chart(df).mark_boxplot().encode(
        y='SuS-O-Meter:Q'
    ).properties(
        title='Boxplot des SuS-O-Meters'
    )

    chart.save(output_path / 'susometer.html')
    chart_stand.save(output_path / 'susometer_stand.html')
    box.save(output_path / 'susometer_box.html')


    # Erstellen des Zwischenpfads und Speichern der Datei
    interim_path.mkdir(parents=True, exist_ok=True)

    df_assistance = df_assistance.convert_dtypes()
    df_assistance.to_csv(interim_path / 'assistance.csv', index=False)

    logger.info('Prepare assistance report ... done')
    logger.info('Prepare workshop file ...')

    # Read and prepare workshop file
    df_workshop = pd.read_excel(open(input_path / 'Q-Lines_anonymized.xlsx', 'rb'))
    df_workshop = df_workshop.convert_dtypes()
    df_workshop['Reparaturbeginndatum'] = pd.to_datetime(df_workshop['Reparaturbeginndatum'], format='%Y%m%d')

    # Drop duplicated Q-Line entries (623) -> in Q&A4 abgenommen
    df_workshop = df_workshop.drop_duplicates(subset=['Q-Line'])

    # Rename FIN in VIN
    df_workshop = df_workshop.rename(columns={'FIN': 'VIN'})

    # Sort the DataFrame by VIN and repair start date and calculate the difference between repair start dates
    df_workshop = df_workshop.sort_values(by=['VIN', 'Reparaturbeginndatum']).reset_index(drop=True)
    df_workshop['Time_Diff'] = df_workshop.groupby('VIN')['Reparaturbeginndatum'].diff().dt.days

    # Identify where the workshop stay is the same or the difference in days is <= 7
    df_workshop['Same_Werkstattaufenthalt'] = df_workshop['Werkstattaufenthalt'] == df_workshop.groupby('VIN')[
        'Werkstattaufenthalt'].shift()
    df_workshop['Within_7_Days'] = df_workshop['Time_Diff'] <= 7

    # Replace NA values with False
    df_workshop['Same_Werkstattaufenthalt'] = df_workshop['Same_Werkstattaufenthalt'].fillna(False)
    df_workshop['Within_7_Days'] = df_workshop['Within_7_Days'].fillna(False)

    # Identify new stays based on the conditions
    df_workshop['New_Stay'] = ~(df_workshop['Same_Werkstattaufenthalt'] | df_workshop['Within_7_Days'])

    # Keep track of VIN changes
    df_workshop['VIN_Change'] = df_workshop['VIN'] != df_workshop['VIN'].shift()

    # New stays are also marked by a VIN change
    df_workshop['New_Stay'] = df_workshop['New_Stay'] | df_workshop['VIN_Change']

    # Calculate the Aufenthalt_ID by cumulatively summing the new stays
    df_workshop['Aufenthalt_ID'] = df_workshop.groupby('VIN')['New_Stay'].cumsum()

    # Remove the temporary columns
    df_workshop = df_workshop.drop(
        columns=['Time_Diff', 'Same_Werkstattaufenthalt', 'Within_7_Days', 'New_Stay', 'VIN_Change'])

    # Add VIN to Aufenthalt_ID
    df_workshop['Aufenthalt_ID'] = df_workshop['VIN'] + '_' + df_workshop['Aufenthalt_ID'].astype('string')

    df_workshop.convert_dtypes()
    df_workshop.to_csv(interim_path / 'workshop.csv', index=False)

    logger.info('Prepare workshop file ... done')
    logger.info('Start matching files ...')

    df_assistance_filtered = df_assistance[
        df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])].copy()

    df_assistance_filtered['Incident Date Datum'] = df_assistance_filtered['Incident Date'].dt.normalize()

    df_assistance_filtered = df_assistance_filtered.sort_values(by=['Incident Date Datum', 'VIN']).reset_index(
        drop=True)
    df_workshop = df_workshop.sort_values(by=['Reparaturbeginndatum', 'VIN']).reset_index(drop=True)
    df_workshop_filtered = df_workshop.drop_duplicates(subset=['Aufenthalt_ID'])

    # Create a plot to see how many tolerance days are good to keep
    # We decided on 7 after Porsche told us in Q&A4 that 5 days is acceptable, 7 days is with 2 days of buffer
    # This is merge is only on assistance calls where the outcome description is towing or scheduled towing and only on
    # workshop entries which are the first for our business logic of condensed repairs (Aufenthalt_ID)
    results = []
    for tolerance in range(1, 31):
        tmp_merge = pd.merge_asof(df_assistance_filtered, df_workshop_filtered, left_on='Incident Date Datum',
                                  right_on='Reparaturbeginndatum', by='VIN', direction='forward',
                                  tolerance=pd.Timedelta(days=tolerance))
        num_merges = len(tmp_merge[~tmp_merge['Aufenthalt_ID'].isna()])

        results.append({'Toleranz in Tagen': tolerance, 'Anzahl an Merges': num_merges})

    results_df = pd.DataFrame(results)

    merges_chart = alt.Chart(results_df).mark_bar().encode(
        x=alt.X('Toleranz in Tagen'),
        y=alt.Y('Anzahl an Merges:Q', scale=alt.Scale(domainMax=len(df_assistance_filtered))),
        tooltip=['Toleranz in Tagen', 'Anzahl an Merges']
    )

    merges_chart.save(output_path / 'num_merges.html')

    merged_on_towing_df = pd.merge_asof(df_assistance_filtered, df_workshop_filtered, left_on='Incident Date Datum',
                                        right_on='Reparaturbeginndatum', by='VIN', direction='forward',
                                        tolerance=pd.Timedelta(days=7))

    # There exist entries in the workshop file where it can be assumed that the workshop entry and calls are correlated
    # because of a minimal difference. Here we defined it as three days after an analysis:
    # Example: Many entries in the assistance file where Component, Outcome Description, or Reason of Call are NA (empty)
    # but there is a workshop entry which can be assumed to be correlated
    # Entferne Einträge mit 'Towing' oder 'Scheduled Towing' in der Outcome Description
    df_assistance_no_towing = df_assistance[
        ~df_assistance['Outcome Description'].isin(['Towing', 'Scheduled Towing'])].copy()

    # Erstelle eine neue Spalte mit normalisiertem Datum
    df_assistance_no_towing['Incident Date Datum'] = df_assistance_no_towing['Incident Date'].dt.normalize()

    # Liste der Fall_IDs, die bereits in df_assistance_filtered enthalten sind
    fall_id_with_towing = df_assistance_filtered['Fall_ID'].unique().tolist()

    # Entferne Einträge mit Fall_IDs, die in df_assistance_filtered vorhanden sind
    df_assistance_no_towing_call_before_towing = df_assistance_no_towing[
        df_assistance_no_towing['Fall_ID'].isin(fall_id_with_towing)]
    df_assistance_no_towing = df_assistance_no_towing[~df_assistance_no_towing['Fall_ID'].isin(fall_id_with_towing)]

    # Sortiere und behalte nur den letzten Eintrag für jede Fall_ID
    df_assistance_no_towing = df_assistance_no_towing.sort_values(by=['Incident Date', 'VIN'])
    df_assistance_no_towing_only_last_call = df_assistance_no_towing.drop_duplicates(subset='Fall_ID', keep='last')

    # Die restlichen Einträge in ein separates DataFrame speichern
    tmp_df_assistance_no_towing = df_assistance_no_towing[
        ~df_assistance_no_towing.index.isin(df_assistance_no_towing_only_last_call.index)]

    # Zurücksetzen des Index für saubere Verarbeitung später
    df_assistance_no_towing = df_assistance_no_towing_only_last_call.sort_values(
        by=['Incident Date Datum', 'VIN']).reset_index(drop=True)

    results_not_on_towing = []
    for tolerance in range(0, 10):
        tmp_merges_not_on_towing = pd.merge_asof(df_assistance_no_towing, df_workshop_filtered,
                                                 left_on='Incident Date Datum',
                                                 right_on='Reparaturbeginndatum', by='VIN', direction='forward',
                                                 tolerance=pd.Timedelta(days=tolerance))
        num_merges = len(tmp_merges_not_on_towing[~tmp_merges_not_on_towing['Aufenthalt_ID'].isna()])

        results_not_on_towing.append({'Toleranz in Tagen': tolerance, 'Anzahl an Merges': num_merges})

    results_not_on_towing_df = pd.DataFrame(results_not_on_towing)

    merges_not_on_towing_chart = alt.Chart(results_not_on_towing_df).mark_bar().encode(
        x=alt.X('Toleranz in Tagen'),
        y=alt.Y('Anzahl an Merges'),
        tooltip=['Toleranz in Tagen', 'Anzahl an Merges']
    )

    merges_not_on_towing_chart.save(output_path / 'num_merges_not_on_towing.html')

    merged_not_on_towing_df = pd.merge_asof(df_assistance_no_towing, df_workshop_filtered,
                                            left_on='Incident Date Datum',
                                            right_on='Reparaturbeginndatum', by='VIN', direction='forward',
                                            tolerance=pd.Timedelta(days=3))

    merged_df = pd.concat([merged_on_towing_df, df_assistance_no_towing_call_before_towing, merged_not_on_towing_df,
                           tmp_df_assistance_no_towing], ignore_index=True)

    fall_id_to_aufenthalt_id = merged_df[['Fall_ID', 'Aufenthalt_ID']].copy()
    fall_id_to_aufenthalt_id = fall_id_to_aufenthalt_id.dropna()

    merged_df.convert_dtypes()
    merged_df.to_csv(interim_path / 'merged.csv', index=False)
    fall_id_to_aufenthalt_id.to_csv(interim_path / 'fall_id_to_aufenthalt_id.csv', index=False)

    logger.info('Matched files ... Done')

    # Laden der vorbereiteten Daten
    # df_assistance_filtered = pd.read_csv('data/interim/assistance.csv')
    # df_workshop = pd.read_csv('data/interim/workshop.csv')
    # fall_id_to_aufenthalt_id = pd.read_csv('data/interim/fall_id_to_aufenthalt_id.csv')

    # Filter für "Towing" oder "Scheduled Towing" in der Assistance-Datei
    df_assistance_filtered_towing = df_assistance_filtered[
        df_assistance_filtered['Outcome Description'].isin(['Towing', 'Scheduled Towing'])
    ]

    # Liste der Fall_IDs, die gemerged wurden
    merged_fall_ids = fall_id_to_aufenthalt_id['Fall_ID'].unique()

    # Fall-IDs, die nicht gemerged wurden
    unmerged_fall_ids = df_assistance_filtered_towing[~df_assistance_filtered_towing['Fall_ID'].isin(merged_fall_ids)]

    # Ergebnis-DataFrame
    unmerged_fall_ids_df = unmerged_fall_ids[['Fall_ID', 'VIN', 'Incident Date', 'Outcome Description']]

    # Ausgabe des DataFrames
    unmerged_fall_ids_df.to_csv('data/interim/unmerged_fall_ids.csv', index=False)

    # Neue Spalte in df_workshop erstellen und mit 'yes'/'no' füllen
    df_workshop['Unmerged_Towing_STowing_SUS'] = df_workshop['Aufenthalt_ID'].isin(unmerged_fall_ids_df['Fall_ID']).map(
        {True: 'True', False: 'False'})

    # Speichern der aktualisierten workshop.csv
    df_workshop.to_csv('data/interim/workshop.csv', index=False)

    # Anzahl der Einträge in unmerged_fall_ids_df ausgeben
    num_unmerged_entries = len(unmerged_fall_ids_df)
    print(f"Anzahl der Einträge in unmerged_fall_ids_df: {num_unmerged_entries}")

    # ToDo
    # Überpürfen ob Reparaturdaten bei Werkstattaufenthalten identisch sind (Marcs Idee)
