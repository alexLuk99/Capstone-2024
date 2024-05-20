from loguru import logger
from pathlib import Path

import pandas as pd


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

    # Convert Incident Date to DateTime and add time from Time of Call

    df_assistance['Incident Date'] = pd.to_datetime(df_assistance['Incident Date'], format='%d/%m/%Y')
    df_assistance['Day of Incident'] = df_assistance['Incident Date']
    df_assistance['Incident Date'] = df_assistance['Incident Date'] + pd.to_timedelta(
        df_assistance['Time Of Call'] + ':00')

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
        '7': '',
        '8': ''
    }

    df_assistance['Reason Of Call Mapped'] = df_assistance['Reason Of Call'].map(mapping_roc)
    df_assistance.loc[df_assistance['Reason Of Call Mapped'].isna(), 'Reason Of Call Mapped'] = df_assistance[
        'Reason Of Call']
    df_assistance = df_assistance.drop(columns=['Reason Of Call'])
    df_assistance = df_assistance.rename(columns={'Reason Of Call Mapped': 'Reason Of Call'})

    # Odomoter aufbereiten, negative Werte, Werte unter 100km, und Werte über 100.000km werden mit pd.NA ersetzt
    df_assistance['Odometer'] = pd.to_numeric(df_assistance['Odometer'], errors='coerce')
    df_assistance.loc[(df_assistance['Odometer'] <= 100) | (df_assistance['Odometer'] >= 999_999), 'Odometer'] = pd.NA

    # Anrufe vom gleichen Fahrzeug am gleichen Tag werden nur einmal mitgenommen
    df_assistance = df_assistance.sort_values(by='Incident Date')
    df_assistance_filtered = df_assistance.drop_duplicates(subset=['VIN', 'Day of Incident'])

    # df_assistance_filtered['14_day_window'] = df_assistance_filtered.groupby('VIN')['Incident Date'].diff().dt.days <= 14
    # multiple_calls = df_assistance_filtered[df_assistance_filtered['14_day_window']].groupby('VIN').size().reset_index(name='counts')
    # multiple_calls = multiple_calls[multiple_calls['counts'] > 1]

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

    df_workshop.convert_dtypes()
    df_workshop.to_csv(interim_path / 'workshop.csv', index=False)

    filtered_df.convert_dtypes()
    filtered_df.to_csv('data/interim/filtered.csv', index=False)


    logger.info('Prepare workshop file ... done')

    # Merge files if possible (don't think we can tbh)

    # Return either single file or both in a dataclass
