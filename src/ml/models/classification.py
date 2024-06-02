import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

def classification(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame) -> None:
    # 'VIN' als Index setzen
    df_assistance.set_index('VIN', inplace=True)

    # Gruppieren und Zählen der Reparaturen pro 'VIN'
    df_repairs = df_workshop.groupby('VIN')['Q-Line'].count().rename('Anzahl Q-Line')

    # Merge der Reparaturen in df_assistance
    df_assistance = df_assistance.merge(df_repairs, left_index=True, right_index=True, how='left')
    df_assistance['Anzahl Q-Line'].fillna(0, inplace=True)  # Fehlende Werte auffüllen

    # Features und Labels trennen
    X = df_assistance[['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component', 'Outcome Description', 'RSA Successful', 'Anzahl Q-Line']]
    y = df_assistance['Merged']

    # Definieren der Transformer für kategorische und numerische Daten
    categorical_columns = ['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component', 'Outcome Description', 'RSA Successful']
    numerical_columns = ['Anzahl Q-Line']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer zur Anwendung der jeweiligen Transformer auf die entsprechenden Spalten
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns)
        ])

    # Pipeline für den gesamten Prozess
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42))
    ])

    # Hyperparameter-Raster
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    }

    # GridSearchCV Initialisieren
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter-Tuning und Modell-Training
    grid_search.fit(X_train, y_train)

    # Beste Parameter und bestes Modell
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Beste Parameter: {best_params}")

    # Vorhersagen treffen
    y_pred = best_model.predict(X_test)

    # Modellbewertung
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Feature-Wichtigkeit anzeigen
    model = best_model.named_steps['classifier']
    feature_importances = model.feature_importances_
    feature_names = best_model.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names_out(categorical_columns)
    importances = pd.Series(feature_importances, index=feature_names)
    print(importances.sort_values(ascending=False))