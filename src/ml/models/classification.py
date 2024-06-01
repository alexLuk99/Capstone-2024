import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def classification(df_assistance: pd.DataFrame) -> None:
    # 'VIN' als Index setzen
    df_assistance.set_index('VIN', inplace=True)

    # Features und Labels trennen
    X = df_assistance[
        ['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component', 'Outcome Description',
         'RSA Successful']]
    y = df_assistance['Merged']

    # Kategorische Daten mit LabelEncoder kodieren
    categorical_columns = ['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component',
                           'Outcome Description', 'RSA Successful']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Sicherstellen, dass alle Daten numerisch sind
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(float)
            except ValueError:
                print(f"Spalte {col} enthält nicht-numerische Werte, die nicht umgewandelt werden konnten.")
                print(X[col].unique())
                return

    # Fehlende Werte imputieren (falls vorhanden)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell initialisieren und trainieren
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = model.predict(X_test)

    # Modellbewertung
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Feature-Wichtigkeit anzeigen
    feature_importances = model.feature_importances_
    features = X.columns
    importances = pd.Series(feature_importances, index=features)
    print(importances)


# Beispielaufruf
if __name__ == "__main__":
    # Lade den DataFrame (ersetze den Pfad durch deinen tatsächlichen Pfad)
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)

    # Aufruf der Klassifikationsfunktion
    classification(df_assistance)
