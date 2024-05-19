import pandas as pd
from scipy.stats import chi2_contingency
import altair as alt


def perform_chi_square_test(data: pd.DataFrame, col1: str, col2: str) -> None:
    """
    Performs a chi-squared test of independence between two categorical variables in a DataFrame.

    This function calculates the observed and expected frequencies, as well as the residuals for the chi-squared test,
    and returns a DataFrame that combines these results.
    :param data: The input DataFrame containing the categorical data.
    :param col1: The name of the first categorical column.
    :param col2: The name of the second categorical column.
    :return: None

    Notes
    -----
    - The function removes any rows in the input DataFrame that contain NaN values in `col1` or `col2`.
    - The chi-squared test assumes that the observed and expected frequencies are sufficiently large for the test to be valid.

    """
    df = data[[col1, col2]].copy()
    df = df.dropna()

    confusion_matrix = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    residuals = confusion_matrix - expected
    significant = p < 0.05  # 5% significance level

    # Create a DataFrame for observed, expected, and residuals
    observed_df = confusion_matrix.reset_index().melt(id_vars=confusion_matrix.index.name)
    expected_df = pd.DataFrame(expected, index=confusion_matrix.index,
                               columns=confusion_matrix.columns).reset_index().melt(id_vars=confusion_matrix.index.name)
    residuals_df = residuals.reset_index().melt(id_vars=residuals.index.name)

    observed_df = observed_df.rename(columns={'value': 'Observed'})
    expected_df = expected_df.rename(columns={'value': 'Expected'})
    residuals_df = residuals_df.rename(columns={'value': 'Residuals'})

    result_df = pd.concat([observed_df, expected_df['Expected'], residuals_df['Residuals']], axis=1)

    counts = _get_count_chart(data=df, col1=col1, col2=col2)
    visualize_chi_square_test(data=result_df, col1=col1, col2=col2, chi2=chi2, p=p, significant=significant, dof=dof,
                              counts=counts)


def visualize_chi_square_test(data: pd.DataFrame, col1: str, col2: str, chi2: float, p: float, significant: bool,
                              dof: int, counts: alt.Chart) -> None:
    """
    Takes measures calculated in perform_chi_square_test() and plots the chi-squared test results as a heatmap.
    :param data: Input DataFrame containing the categorical data, observed, expected, and residual frequencies.
    :param col1: The categories of the first variable.
    :param col2: The categories of the second variable.
    :param chi2: Chi2 value for the chi-squared test.
    :param p: The p-value for the chi-squared test.
    :param significant: Whether the chi-squared test is significant. (5% significant level)
    :param dof: Degrees of freedom for the chi-squared test.
    :return: None (alt.Chart)
    """
    cols = ['Observed', 'Expected', 'Residuals']
    charts = []

    for column in cols:
        _tmp1, _tmp2 = [col for col in cols if col != column]

        heatmap = alt.Chart(data).mark_rect().encode(
            x=f'{col1}:O',
            y=f'{col2}:O',
            color=alt.Color(f'{column}:Q', scale=alt.Scale(scheme='blues')),
            tooltip=[f'{col1}', f'{col2}', f'{column}', f'{_tmp1}', f'{_tmp2}']
        ).properties(
            title={
                'text': f'{column} Frequencies',
                'subtitle': f'Chi2: {chi2:.2f} | p-Value: {p:.2f} | Significant: {significant} | Degree of Freedom: {dof}'
            },
            width=600,
            height=600
        )

        charts.append(heatmap)

    chart = alt.vconcat(
        alt.hconcat(counts, charts[0]),
        alt.hconcat(charts[1], charts[2]),
    )

    chart.save(f'output/{col1}_{col2}_{column}.html')


def _get_count_chart(data: pd.DataFrame, col1: str, col2: str) -> alt.Chart:
    """
    Creates a simple heatmap showing the correlation between column 1 and column 2.
    :param data: The input DataFrame containing the categorical data.
    :param col1: The categories of the first variable.
    :param col2: The categories of the second variable.
    :return: alt.Chart
    """
    counts_df = data.value_counts().reset_index()
    counts = alt.Chart(counts_df).mark_rect().encode(
        x=alt.X(f'{col1}:N', title=f'{col1}'),
        y=alt.Y(f'{col2}:N', title=f'{col2}'),
        color=alt.Color(f'count:Q', title=f'Anzahl'),
        tooltip=[
            alt.Tooltip(title=f'{col1}', field=f'{col1}'),
            alt.Tooltip(title=f'{col2}', field=f'{col2}'),
            alt.Tooltip(title=f'Anzahl', field=f'count'),
        ]
    ).properties(
        title={
            'text': f'{col1} - {col2} Anzahl',
        },
        width=600,
        height=600
    )

    return counts
