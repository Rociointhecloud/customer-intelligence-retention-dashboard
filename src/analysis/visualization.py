import pandas as pd
import matplotlib.pyplot as plt


def plot_segment_distribution(df: pd.DataFrame) -> None:
    counts = df["segment_name"].value_counts().sort_values(ascending=False)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Customer Distribution by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_revenue_by_segment(df: pd.DataFrame) -> None:
    revenue = (
        df.groupby("segment_name")["monetary_total"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure()
    revenue.plot(kind="bar")
    plt.title("Total Revenue by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_churn_rate_by_segment(df: pd.DataFrame) -> None:
    churn_col = [c for c in df.columns if c.startswith("churn_")][0]

    churn_rate = (
        df.groupby("segment_name")[churn_col]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    plt.figure()
    churn_rate.plot(kind="bar")
    plt.title("Churn Rate (%) by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()