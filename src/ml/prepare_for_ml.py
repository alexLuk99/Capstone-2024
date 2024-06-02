import os
from pathlib import Path

import pandas as pd
from loguru import logger

from sklearn.preprocessing import LabelEncoder
from joblib import dump
from config.paths import models_path

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


def prepare_for_ml(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    logger.info('Data preparation for clustering and classification')
    label_encoder_path = Path(models_path / 'label_encoders.joblib')

    # Create dataframe with most important information
    cols = ['Telephone Help', 'Service Paid By Customer']

    for col in cols:
        df_assistance[col] = df_assistance[col].fillna('NO')
        df_assistance[col] = df_assistance[col].str.upper()
        df_assistance[col] = df_assistance[col].map({'YES': True, 'NO': False}).astype(bool)
        df_assistance[col] = df_assistance[col].apply(lambda x: x if isinstance(x, bool) else False)

    df_assistance['Rental Car Days'] = df_assistance['Rental Car Days'].fillna(0)
    df_assistance['days_since_last_towing'] = df_assistance['days_since_last_towing'].fillna(0)

    df_assistance_grouped = df_assistance.groupby(by='VIN', as_index=False).agg(
        Anzahl_Anrufe=pd.NamedAgg(column='Incident Date', aggfunc='count'),
        Durschnittliche_Zeit_zwischen_Towings=pd.NamedAgg(column='days_since_last_towing', aggfunc='mean'),
        Telephone_Help=pd.NamedAgg(column='Telephone Help', aggfunc='sum'),
        Service_Paid_By_Customer=pd.NamedAgg(column='Service Paid By Customer', aggfunc='sum'),
        Rental_Car=pd.NamedAgg(column='Rental Car', aggfunc='sum'),
        Rental_Car_Days=pd.NamedAgg(column='Rental Car Days', aggfunc='sum'),
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

    df_assistance_grouped['Total Services Offered'] = df_assistance_grouped[cols].sum(axis=1)
    df_assistance_grouped = df_assistance_grouped.drop(columns=cols)

    df_assistance = df_assistance.sort_values(by=['Registration Date', 'VIN'])
    df_assistance_first_registration_date = df_assistance[['VIN', 'Registration Date']].drop_duplicates(subset=['VIN'])
    df_assistance_first_registration_date['Registration Date'] = pd.to_datetime(
        df_assistance_first_registration_date['Registration Date'])
    df_assistance_first_registration_date['Registration Date Jahr'] = df_assistance_first_registration_date[
        'Registration Date'].dt.year
    df_assistance_first_registration_date = df_assistance_first_registration_date.drop(columns='Registration Date')

    df_assistance_grouped = df_assistance_grouped.merge(df_assistance_first_registration_date, on='VIN', how='left')

    # Label Encoding der Spalte Modellreihe
    label_encoder_modellreihe = LabelEncoder()
    df_assistance['Modellreihe'] = df_assistance['Modellreihe'].fillna('Nicht definiert')
    df_assistance['Modellreihe_Encoded'] = label_encoder_modellreihe.fit_transform(df_assistance['Modellreihe'])

    tmp_modellreihe = df_assistance[['VIN', 'Modellreihe_Encoded']].copy().drop_duplicates(subset='VIN')
    df_assistance_grouped = df_assistance_grouped.merge(tmp_modellreihe, on='VIN')

    # Save the label encoders
    dump(label_encoder_modellreihe, label_encoder_path)

    # Spalten als features hinzufügen
    cols_to_pivot = ['Outcome Description', 'Component']

    for col in cols_to_pivot:
        tmp = df_assistance.pivot_table(index='VIN', columns=col, aggfunc='size', fill_value=0).reset_index()
        df_assistance_grouped = df_assistance_grouped.merge(tmp, on='VIN', how='left')
        df_assistance_grouped = df_assistance_grouped.fillna(0)

    # Anzahl Reparaturen pro VIN
    df_repairs = df_workshop.groupby(by='VIN', as_index=False).agg(
        Repairs=pd.NamedAgg(column='Q-Line', aggfunc='count'),
    )

    df_aufenthalte = df_workshop.groupby(by='VIN', as_index=False).agg(
        Aufenthalte=pd.NamedAgg(column='Werkstattaufenthalt', aggfunc='nunique')
    )

    df_assistance_grouped = df_assistance_grouped.merge(df_repairs, on='VIN', how='left')
    df_assistance_grouped = df_assistance_grouped.merge(df_aufenthalte, on='VIN', how='left')

    # Set VIN as Index
    data = df_assistance_grouped.copy()

    # Suspect hinzufügen
    df_suspect = df_assistance[['VIN', 'Suspect']].drop_duplicates(subset='VIN')

    data_supsect = data.copy()
    data_supsect = data_supsect.merge(df_suspect, on='VIN', how='left')

    data = data.set_index('VIN')
    data_supsect = data_supsect.set_index('VIN')

    return data, data_supsect
