from datetime import timedelta

import re
import numpy as np
from loguru import logger
from pathlib import Path
import numpy as np

import pandas as pd


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
        tmp = pd.read_excel(open('data/raw/Assistance_Report_Europa_2021-2023_anonymized.xlsx', 'rb'), sheet_name=sheet)
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

    # replace CTA with our calculated difference between Time Of Arrival and Incident Date
    df_assistance['CTA'] = df_assistance['Time Of Arrival'] - df_assistance['Incident Date']
    df_assistance.loc[df_assistance['CTA'] <= pd.Timedelta(0), 'CTA'] = pd.NaT
    df_assistance['CTA'] = (df_assistance['CTA'].dt.total_seconds() // 60).fillna(0).astype('Int64').replace(0, pd.NA)

    # todo
    #   check again -> many values can not be processed
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
    df_assistance.loc[df_assistance['Policy Start Date'] < pd.to_datetime('2002-01-01', format='%Y-%m-%d'), 'Policy Start Date'] = pd.NaT
    df_assistance.loc[df_assistance['Policy End Date'] >= pd.to_datetime('2027-01-01', format='%Y-%m-%d'), 'Policy End Date'] = pd.NaT
    df_assistance.loc[df_assistance['Policy End Date'] <= df_assistance['Policy Start Date'], 'Policy End Date'] = pd.NaT

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

    # create interim path
    interim_path = Path('data/interim')
    interim_path.mkdir(parents=True, exist_ok=True)

    # Write processed assistance file
    df_assistance = df_assistance.convert_dtypes()
    df_assistance.to_csv(interim_path / 'assistance.csv', index=False)

    logger.info('Prepare assistance report ... done')
    logger.info('Prepare workshop file ...')

    # Read and prepare workshop file
    df_workshop = pd.read_excel(open('data/raw/Q-Lines_anonymized.xlsx', 'rb'))
    df_workshop = df_workshop.convert_dtypes()
    df_workshop['Reparaturbeginndatum'] = pd.to_datetime(df_workshop['Reparaturbeginndatum'], format='%Y%m%d')

    # Konvertiere Datumsangaben in datetime-Objekte
    df_workshop['Reparaturbeginndatum'] = pd.to_datetime(df_workshop['Reparaturbeginndatum'])

    df_workshop.convert_dtypes()
    df_workshop.to_csv(interim_path / 'workshop.csv', index=False)

    logger.info('Prepare workshop file ... done')
    logger.info('Start matching files ...')

    # filtered_df.convert_dtypes()
    # filtered_df.to_csv('data/interim/filtered.csv', index=False)

    # Merging Assistance Workshop

    # Mergen der DataFrames basierend auf VIN und FIN
    merged_df = pd.merge(df_assistance, df_workshop, left_on='VIN', right_on='FIN',
                         suffixes=('_df_assistance', '_df_workshop'))

    # Anwenden der Toleranzbedingungen
    tolerance_days = 14  # 2 Wochen
    tolerance_km = 100

    matched_df = merged_df[
        (abs(merged_df['Incident Date'] - merged_df['Reparaturbeginndatum']) <= timedelta(days=tolerance_days)) &
        (abs(merged_df['Odometer'] - merged_df['Kilometerstand Reparatur']) <= tolerance_km)
        ]

    matched_df.convert_dtypes()
    matched_df.to_csv('data/interim/matched.csv', index=False)

    logger.info('Matched files ... Done')

    # ToDo
    # Überpürfen ob Reparaturdaten bei Werkstattaufenthalten identisch sind (Marcs Idee)
