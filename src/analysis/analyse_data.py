from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta


def analyse_data() -> None:
    # Option 1: Load your existing data
    df_assistance = pd.read_csv("data/interim/assistance.csv")
    df_assistance = df_assistance.convert_dtypes()
    print(df_assistance)

    # Convert date columns, handling errors and filtering out invalid dates
    for col in ["Incident Date", "Policy End Date", "Policy Start Date"]:  # Added Policy Start Date to the loop
        if col == "Incident Date":
            format = "%Y-%m-%d %H:%M:%S"
        elif col == "Policy End Date" or col == 'Policy Start Date':
            format = "%Y-%m-%d"

        df_assistance[col] = pd.to_datetime(
            df_assistance[col], format=format, errors="coerce"
        )
        df_assistance.dropna(subset=[col], inplace=True)

    print(df_assistance)

    # Calculate difference in days, filtering out negative/zero values (same for both options)
    df_assistance["Days Until Policy End"] = (
        df_assistance["Policy End Date"] - df_assistance["Incident Date"]
    ).dt.days
    df_assistance_filtered = df_assistance[
        (df_assistance["Days Until Policy End"] > 0)
        & (df_assistance["Days Until Policy End"] <= 730)
    ]

    # Display the first 5 rows after processing to check (optional)
    print(
        df_assistance_filtered[
            ["Incident Date", "Policy End Date", "Days Until Policy End"]
        ]
        .head()
        .to_markdown(index=False, numalign="left", stralign="left")
    )

    # Regression analysis
    X = sm.add_constant(df_assistance_filtered["Days Until Policy End"])
    y = df_assistance_filtered.index  # Number of incidents is represented by row number

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Display regression results
    print(model.summary())

    # Generate predictions for the regression line
    predictions = model.predict(X)

    # Create a DataFrame for the regression line data
    regression_data = pd.DataFrame(
        {
            "Days Until Policy End": df_assistance_filtered["Days Until Policy End"],
            "Predicted Incidents": predictions,
        }
    )

    # Create the bar chart
    bar_chart = (
        alt.Chart(df_assistance_filtered)
        .mark_bar()
        .encode(
            x=alt.X("Days Until Policy End:Q", title="Days Until Policy End"),
            y=alt.Y("count()", title="Number of Incidents"),
            tooltip=["Days Until Policy End", "count()"],
        )
        .properties(
            title="Distribution of Incidents Relative to Policy End Date",
            width=800,  # Breite in Pixel
            height=400,  # Höhe in Pixel
        )
    )

    # Create the scatter plot with regression line
    scatter_chart = (
        alt.Chart(df_assistance_filtered)
        .mark_circle(size=60)
        .encode(
            x=alt.X("Days Until Policy End:Q", title="Days Until Policy End"),
            y=alt.Y("count()", title="Number of Incidents"),
            tooltip=["Days Until Policy End", "count()"],
        )
    )

    regression_line = (
        alt.Chart(regression_data)
        .mark_line(color="red")
        .encode(
            x="Days Until Policy End:Q",
            y="Predicted Incidents:Q",
        )
    )

    # Save the charts separately
    bar_chart.interactive().save("output/incidents_vs_policy_end_histogram.html")
    regression_line.properties(
        title="Scatterplot of Incidents vs. Policy End Date (with Regression)",
        width=800,  # Breite in Pixel
        height=400,  # Höhe in Pixel
    ).interactive().save("output/incidents_vs_policy_end_scatter_regression.html")

    # Calculate days since policy start
    df_assistance_filtered["Days Since Policy Start"] = (
        df_assistance_filtered["Incident Date"] - df_assistance_filtered["Policy Start Date"]
    ).dt.days

    # Filter for the first year of the policy
    df_assistance_filtered_start = df_assistance_filtered[
        (df_assistance_filtered['Days Since Policy Start'] >= 0) &
        (df_assistance_filtered['Days Since Policy Start'] <= 730)
        ]

    # Check if there are enough rows for regression calculation
    if len(df_assistance_filtered_start) >= 2:
        # Regression analysis
        X = sm.add_constant(df_assistance_filtered_start["Days Since Policy Start"])
        y = df_assistance_filtered_start.index  # Number of incidents is represented by row number

        # Fit the regression model
        model = sm.OLS(y, X).fit()

        # Display regression results
        print(model.summary())

        # Generate predictions for the regression line
        predictions = model.predict(X)

        # Create a DataFrame for the regression line data
        regression_data_start = pd.DataFrame(
            {
                "Days Since Policy Start": df_assistance_filtered_start[
                    "Days Since Policy Start"
                ],
                "Predicted Incidents": predictions,
            }
        )

        # Create the base scatter plot
        scatter_chart_start = (
            alt.Chart(df_assistance_filtered_start)
            .mark_circle(size=60)
            .encode(
                x=alt.X("Days Since Policy Start:Q", title="Days Since Policy Start"),
                y=alt.Y("count()", title="Number of Incidents"),
                tooltip=["Days Since Policy Start", "count()"],
            )
        )

        # Create the regression line
        regression_line_start = (
            alt.Chart(regression_data_start)
            .mark_line(color="red")
            .encode(
                x="Days Since Policy Start:Q",
                y="Predicted Incidents:Q",
            )
        )

        # Combine the scatter plot and regression line
        chart_start = scatter_chart_start + regression_line_start

        # Customize the chart
        chart_start = chart_start.properties(
            title="Scatter Plot with Regression Line (First 12 Months of Policy)",
            width=800,
            height=400,
        ).interactive()

        # Save the chart
        chart_start.save("output/incidents_vs_policy_start_scatter_regression2.html")
    else:
        print("Not enough data to perform regression analysis.")

    df_workshop = pd.read_csv('data/interim/workshop.csv')

    pass
