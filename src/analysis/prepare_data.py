import pandas as pd

def read_prepare_data() -> pd.DataFrame:
    # Read and prepare assistance file
    sheets = ['2021', '2022', '2023']
    assistance_list = []
    for sheet in sheets:
        tmp = pd.read_excel('data/raw/Assistance_Report_Europa_2021-2023_anonymized.xlsx', sheet_name=sheet)
        assistance_list.append(tmp)
    df_assistance = pd.concat(assistance_list, ignore_index=True)
    df_assistance = df_assistance.convert_dtypes()
    df_assistance['Incident Date'] = pd.to_datetime(df_assistance['Incident Date'], format='%d/%m/%Y')
    df_assistance['Incident Date'] = df_assistance['Incident Date'] + pd.to_timedelta(
        df_assistance['Time Of Call'] + ':00')

    date_columns = ['Registration Date', 'Policy Start Date', 'Policy End Date']
    for column in date_columns:
        df_assistance[column] = pd.to_datetime(df_assistance[column], errors='coerce') #coerce -> alle ungültigen Werte werden auf NaT gesetzt

    # Read and prepare workshop file
    df_workshop = pd.read_excel('data/raw/Q-Lines_anonymized.xlsx')
    df_workshop = df_workshop.convert_dtypes()
    df_workshop['Reparaturbeginndatum'] = pd.to_datetime(df_workshop['Reparaturbeginndatum'], format='%d/%m/%Y')

    # Merge files if possible (don't think we can tbh)

    # Return either single file or both in a dataclass


