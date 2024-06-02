import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def classify_and_evaluate(df):
    """
    Diese Funktion führt eine Klassifikation auf dem gegebenen DataFrame durch,
    wobei die letzte Spalte als Zielvariable verwendet wird, die angibt, ob eine
    Fall_ID gemergt wurde oder nicht.

    Parameter:
    df (DataFrame): Ein Pandas DataFrame mit Features und einer Zielspalte.

    Rückgabe:
    None
    """

    # Features und Zielvariable trennen
    X = df.drop(columns=['Merged'])  # Alle Spalten außer der Zielspalte
    y = df['Merged']  # Die Zielspalte

    # Daten in Trainings- und Testset aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Klassifikationsmodell erstellen und trainieren
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = model.predict(X_test)

    # Modellbewertung
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    # Bedeutung der Merkmale anzeigen
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Wichtigkeit der Merkmale')
    plt.xlabel('Wichtigkeit')
    plt.ylabel('Merkmale')
    plt.show()


# Beispielaufruf der Funktion mit einem DataFrame
if __name__ == "__main__":
    # Lade den DataFrame (ersetze den Pfad durch deinen tatsächlichen Pfad)
    df_assistance = pd.read_csv('data/interim/assistance.csv')

    # Überprüfen, ob die Zielspalte "Merged" vorhanden ist
    if 'Merged' in df_assistance.columns:
        # Aufruf der Klassifikationsfunktion
        classify_and_evaluate(df_assistance)
    else:
        print("Die Zielspalte 'Merged' ist nicht im DataFrame vorhanden.")
