import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

def classification(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame, save_path: str) -> None:
    df_assistance.set_index('VIN', inplace=True)
    df_repairs = df_workshop.groupby('VIN')['Q-Line'].count().rename('Anzahl Q-Line')
    df_assistance = df_assistance.merge(df_repairs, left_index=True, right_index=True, how='left')
    df_assistance['Anzahl Q-Line'].fillna(0, inplace=True)

    X = df_assistance[['Country Of Incident', 'Handling Call Center', 'Baureihe', 'Component', 'Outcome Description', 'RSA Successful', 'Anzahl Q-Line']]
    y = df_assistance['Merged']

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

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_columns),
        ('num', numerical_transformer, numerical_columns)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9],
        'classifier__gamma': [0, 0.1, 0.5],
        'classifier__min_child_weight': [1, 5, 10]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Beste Parameter: {best_params}")
    y_pred = best_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

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
    plt.savefig(save_path)
    plt.show()

    model = best_model.named_steps['classifier']
    feature_importances = model.feature_importances_
    categorical_features = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns)
    all_features = list(categorical_features) + numerical_columns
    importances = pd.Series(feature_importances, index=all_features)
    print(importances.sort_values)