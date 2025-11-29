import pandas as pd
import numpy as np


def parse_term(term):
    # useful for sorting by term later for calculating historical averages
    if not isinstance(term, str):
        return 0

    year_str = term[-2:]
    if not year_str.isdigit():
        return 0
    year = int(year_str)
    full_year = 2000 + year

    prefix = term[:-2]

    term_ranks = {"WI": 1, "SP": 2, "S1": 3, "S2": 4, "S3": 5, "SU": 6, "FA": 7}

    rank = term_ranks.get(prefix, 8)

    return full_year * 10 + rank


def calculate_historical_avg(df, group_col, target_col, output_col):
    """
    Calculates the historical average of target_col for each group_col,
    excluding the current term's data.
    """
    valid_mask = df[target_col].notna()
    valid_df = df[valid_mask].copy()

    term_stats = (
        valid_df.groupby([group_col, "term_val"])[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )

    term_stats = term_stats.sort_values([group_col, "term_val"])

    term_stats["cumsum"] = term_stats.groupby(group_col)["sum"].cumsum()
    term_stats["cumcount"] = term_stats.groupby(group_col)["count"].cumsum()

    # excluding current term so it follows the definition of the feature (historical average BEFORE taking the course)
    term_stats["prev_cumsum"] = term_stats.groupby(group_col)["cumsum"].shift(1)
    term_stats["prev_cumcount"] = term_stats.groupby(group_col)["cumcount"].shift(1)

    term_stats[output_col] = term_stats["prev_cumsum"] / term_stats["prev_cumcount"]

    result_df = df.merge(
        term_stats[[group_col, "term_val", output_col]],
        on=[group_col, "term_val"],
        how="left",
    )

    return result_df


def main():
    input_path = "CAPEs_with_TFIDF.tsv"
    df = pd.read_csv(input_path, sep="\t")

    cols_to_clean = ["avg_grade_exp", "avg_grade_rec"]
    for col in cols_to_clean:
        df[col] = df[col].replace(-1.0, np.nan)
    df["term_val"] = df["term"].apply(parse_term)

    df["row_expectation_gap"] = df["avg_grade_exp"] - df["avg_grade_rec"]

    df = calculate_historical_avg(
        df, "instructor", "avg_grade_rec", "Instructor_Historical_Avg_GPA"
    )

    df = calculate_historical_avg(
        df, "sub_course", "avg_grade_rec", "Course_Historical_Avg_GPA"
    )

    df = calculate_historical_avg(
        df, "instructor", "row_expectation_gap", "Instructor_Historical_Expectation_Gap"
    )

    cols_to_drop = ["term_val", "row_expectation_gap"]
    df_final = df.drop(columns=cols_to_drop)

    df_final.to_csv("CAPEs_with_features.csv", index=False)


if __name__ == "__main__":
    main()
