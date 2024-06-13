from pathlib import Path

import pandas as pd
from joblib import dump, load
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config.paths import models_path, output_path


def classification_suspect(data: pd.DataFrame, train_model: bool = False) -> None:
    logger.info('Perform classification Suspect VINs')
    classification_suspect_path = Path(models_path / 'classification_suspect_model.joblib')

    data = data.copy()

    # Total Services Offered, Rental Car Days, Anruferzahl, Towing und Scheduled Towing ist Teil vom Sus-O-Meter -> kommt raus
    data = data.drop(
        columns=['Total Services Offered', 'Anzahl_Anrufe', 'Rental_Car_Days', 'Towing', 'Scheduled Towing'])

    X = data[['Durschnittliche_Zeit_zwischen_Towings', 'Telephone_Help', 'Service_Paid_By_Customer',
              'Registration Date Jahr', 'Modellreihe_Encoded', '(Other)', 'Cancelled',
              'Change of Tyre', 'Jump Start',
              'Rental Car without primary services (i.e. Towing and Roadsaide Assistance)', 'Roadside Repair Others',
              'Scheduled roadside repair', 'Air conditioning ', 'Anti Blocking System (ABS) ', 'Anti Theft Protection',
              'Battery', 'Body-Equipment inside ', 'Brakes - Brake mechanics ',
              'Brakes - Hydraulic brake system, regulator ', 'Clutch', 'Convertible top, hardtop ',
              'Door, central locking system ', 'Engine - Cooling', 'Engine - General',
              'Engine - Lubrication', 'Exhaust system ', 'Final drive - Differential, differential lock ', 'Flat Tyre',
              'Fuel supply', 'Fuel system / Electronic ignition ', 'Generator', 'Glazing, window control ',
              'Ignition and preheating system ', 'Instruments ', 'Insufficient Fuel / Empty Fuel Tank',
              'Key/Lock/Remote Control', 'Level control, air suspension ', 'Lids, flaps ', 'Lights, lamps, switches',
              'Not determinable', 'Passenger protection ', 'Radio, stereo, telephone, on-board computer ',
              'Shift / Selector lever', 'Sliding roof, tilting roof', 'Starter', 'Steering ',
              'Suspension, drive shafts ', 'Transmission', 'Windshield wiper and washer system ', 'Repairs',
              'Aufenthalte']]
    y = data['Suspect']

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if train_model:
        clf = Pipeline(steps=[
            ('classifier', xgb.XGBClassifier(random_state=42))
        ])

        # Hyperparameter-Raster
        param_grid = {
            'classifier__n_estimators': [100, 150],
            'classifier__learning_rate': [0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }

        # GridSearchCV Initialisieren
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Hyperparameter-Tuning und Modell-Training
        grid_search.fit(X_train, y_train)

        # Beste Parameter und bestes Modell
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_

        print(f"Beste Parameter: {best_params}")

        dump(model, classification_suspect_path)

    else:
        if classification_suspect_path.exists():
            model = load(classification_suspect_path)
        else:
            raise FileNotFoundError(f"Model file not found at {classification_suspect_path}")

    # Vorhersagen treffen
    y_pred = model.predict(X_test)

    # Modellbewertung
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ROC-Kurve und AUC berechnen
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    classification_suspect_path = output_path / 'classification_suspect'
    classification_suspect_path.mkdir(exist_ok=True, parents=True)

    # ROC-Kurve plotten und speichern
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(classification_suspect_path / 'classification_suspect_roc.png')
    plt.show()

    classifier = model.named_steps['classifier']
    feature_importances = classifier.feature_importances_
    features = data.columns.tolist()
    features.remove('Suspect')
    importances = pd.Series(feature_importances, index=features)
