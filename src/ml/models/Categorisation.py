from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Annahme: df ist der DataFrame und 'SuS' ist die Spalte, die Anomalien anzeigt
X = df.drop(columns=['SuS'])  # Features
y = df['SuS']  # Zielvariable

# Umwandlung kategorischer Variablen, falls notwendig
X = pd.get_dummies(X)

# Aufteilen in Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren und Trainieren des Modells
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Modellbewertung
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Vorhersagen für neue Daten
new_data = pd.DataFrame({ ... })  # Hier die neuen Daten einfügen
new_data = pd.get_dummies(new_data)
predictions = model.predict(new_data)