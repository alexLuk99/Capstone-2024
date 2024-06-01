import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import altair as alt
from sklearn.preprocessing import StandardScaler


def clustering(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame, df_merged: pd.DataFrame, ) -> None:
    # Create dataframe with most important information

    cols = ['Telephone Help', 'Service Paid By Customer']

    for col in cols:
        df_assistance[col] = df_assistance[col].fillna('NO')
        df_assistance[col] = df_assistance[col].str.upper()
        df_assistance[col] = df_assistance[col].map({'YES': True, 'NO': False}).astype(bool)
        df_assistance[col] = df_assistance[col].apply(lambda x: x if isinstance(x, bool) else False)

    df_assistance['Rental Car Days'] = df_assistance['Rental Car Days'].fillna(0)

    df_assistance_grouped = df_assistance.groupby(by='VIN', as_index=False).agg(
        Anzahl_Anrufe=pd.NamedAgg(column='Incident Date', aggfunc='count'),
        Telephone_Help=pd.NamedAgg(column='Telephone Help', aggfunc='sum'),
        Service_Paid_By_Customer=pd.NamedAgg(column='Service Paid By Customer', aggfunc='sum'),
        Rental_Car=pd.NamedAgg(column='Rental Car', aggfunc='sum'),
        Hotel_Service=pd.NamedAgg(column='Hotel Service', aggfunc='sum'),
        Alternative_Transport=pd.NamedAgg(column='Alternative Transport', aggfunc='sum'),
        Taxi_Service=pd.NamedAgg(column='Taxi Service', aggfunc='sum'),
        Vehicle_Transport=pd.NamedAgg(column='Vehicle Transport', aggfunc='sum'),
        Car_Key_Service=pd.NamedAgg(column='Car Key Service', aggfunc='sum'),
        Parts_Service=pd.NamedAgg(column='Parts Service', aggfunc='sum'),
        Additional_Services_Not_Covered=pd.NamedAgg(column='Additional Services Not Covered', aggfunc='sum')
    )

    cols_to_pivot = ['Modellreihe', 'Country Of Origin', 'Country Of Incident', 'Component', 'Outcome Description']

    for col in cols_to_pivot:
        tmp = df_assistance.pivot_table(index='VIN', columns=col, aggfunc='size', fill_value=0).reset_index()
        df_assistance_grouped = df_assistance_grouped.merge(tmp, on='VIN', how='left')
        df_assistance_grouped = df_assistance_grouped.fillna(0)

    # Set VIN as Index
    data = df_assistance_grouped.copy()
    data = data.set_index('VIN')

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)

    # Applying PCA
    pca = PCA(n_components=features_scaled.shape[1])
    pca.fit(features_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Create a DataFrame for the explained variance
    df_variance = pd.DataFrame({
        'Number of Components': np.arange(1, len(explained_variance) + 1),
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    })

    # Elbow plot using Altair
    elbow_plot = alt.Chart(df_variance).mark_line(point=True).encode(
        x=alt.X('Number of Components:Q', title='Number of Components'),
        y=alt.Y('Explained Variance:Q', title='Explained Variance')
    ).properties(
        title='Elbow Plot of Explained Variance'
    )

    # Cumulative explained variance plot using Altair
    cumulative_plot = alt.Chart(df_variance).mark_line(point=True, color='red').encode(
        x=alt.X('Number of Components:Q', title='Number of Components'),
        y=alt.Y('Cumulative Variance:Q', title='Cumulative Variance')
    ).properties(
        title='Cumulative Explained Variance'
    )

    # Display both plots
    combined_plot = elbow_plot | cumulative_plot
    combined_plot.save('test.html')
