from pathlib import Path

import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

from config.paths import output_path, models_path


def classification(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame, train_model: bool = False) -> None:
    df_assistance.set_index('VIN', inplace=True)
    df_repairs = df_workshop.groupby('VIN')['Q-Line'].count().rename('Anzahl Q-Line')
    df_assistance = df_assistance.merge(df_repairs, left_index=True, right_index=True, how='left')
    df_assistance['Anzahl Q-Line'] = df_assistance['Anzahl Q-Line'].fillna(0)

    classification_suspect_path = Path(models_path / 'classification_model.joblib')

    classification_output = output_path / 'classification'
    classification_output.mkdir(exist_ok=True, parents=True)

    #Auskommentiert 'Outcome Description', 'RSA Successful',

    X = df_assistance[['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component', 'Anzahl Q-Line']]
    y = df_assistance['Merged']

    #Auskommentiert , 'Outcome Description', 'RSA Successful'
    categorical_columns = ['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component']
    numerical_columns = ['Anzahl Q-Line']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if train_model:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns)
        ])

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [3, 5, 7, 9, 12, 15],
        }

        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print(f"Beste Parameter: {best_params}")

        dump(best_model, classification_suspect_path)
    else:
        if classification_suspect_path.exists():
            best_model = load(classification_suspect_path)
        else:
            raise FileNotFoundError(f"Model file not found at {classification_suspect_path}")



    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(classification_output / 'classification_roc.png')

    model = best_model.named_steps['classifier']
    feature_importances = model.feature_importances_
    categorical_features = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns)
    all_features = list(categorical_features) + numerical_columns
    importances = pd.Series(feature_importances, index=all_features)
    print(importances.sort_values)