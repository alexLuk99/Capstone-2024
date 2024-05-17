import pandas as pd
from scipy.stats import chi2_contingency


def perform_chi_square_test(data: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    df = data[[col1, col2]].copy()
    df = df.dropna()

    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    significant = p < 0.05  # 5% significance level

    return chi2, p, significant
