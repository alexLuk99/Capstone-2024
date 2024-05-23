import pandas as pd
from pathlib import Path
import re
from collections import Counter
import altair as alt


def most_common_word(data: pd.DataFrame, output_path: Path) -> None:
    df = data.copy()

    # Fault description Wörter
    # Einlesen der Stopwords aus der CSV-Datei
    stopwords_df = pd.read_csv('utils/stopwords.csv')
    # print("Spaltennamen der Stopwords CSV-Datei:", stopwords_df.columns.tolist())

    # Angenommen, die Stopwords sind in einer Spalte mit dem Namen 'stopwords' enthalten
    stopwords_column_name = 'stopwords'  # Passe dies an, falls die Spalte anders heißt
    stopwords = set(stopwords_df[stopwords_column_name].str.lower())

    # Häufigste Wörter in der Spalte 'Fault Description Customer'
    text = ' '.join(df['Fault Description Customer'].dropna().astype(str))
    words = re.findall(r'\b\w+\b', text.lower())

    # Filterung der Wörter, um Stopwords und Zahlen zu entfernen
    filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(200)

    # Ausgabe der häufigsten Wörter
    # print("Die XX am häufigsten verwendeten Wörter in der Spalte 'Fault Description Customer':")
    # for word, count in most_common_words:
    #     print(f'{word}: {count}')

    # Speichern der häufigsten Wörter als CSV
    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
    common_words_df.to_csv(output_path / 'most_common_words.csv', index=False)

    chart = alt.Chart(common_words_df).mark_bar().encode(
        x=alt.X('Word', sort='-y', title='Wort'),
        y=alt.Y('Count', title='Anzahl'),
        tooltip=['Word', 'Count']
    ).properties(
        title='Häufigste Wörter in der Spalte "Fault Description Customer"',
        width=800,
        height=400
    ).interactive()

    chart.save(output_path / 'most_common_words.html')

    pass
