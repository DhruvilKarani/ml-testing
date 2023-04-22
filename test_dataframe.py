import pandas as pd
import numpy as np
import pytest
import great_expectations as gx


TEST_DF_PATH = "./Iris.csv"

@pytest.fixture(scope="session")
def df():
    df_pandas = pd.read_csv(TEST_DF_PATH)
    df = gx.dataset.PandasDataset(df_pandas)
    return df

@pytest.mark.parametrize("column_name", 
                         ["SepalLengthCm", "SepalWidthCm", 
                          "PetalLengthCm", "PetalWidthCm"])
def test_df_for_na_values(df, column_name):
    # Check for na values
    output = df.expect_column_values_to_not_be_null(column=column_name)
    assert output["success"], f"DataFrame contains na values in {column_name}: {df[column_name].isna().mean()}"


@pytest.mark.parametrize("column_name, lower_limit, upper_limit", 
                         [("SepalLengthCm", 5.2, 6.2), ("SepalWidthCm", 2.5, 3.5), 
                          ("PetalLengthCm", 3.0, 4.0), ("PetalWidthCm", 1.0, 2.0)])
def test_column_mean_range(df, column_name, lower_limit, upper_limit):
    # Check if the mean of a column is within a certain range
    output = df.expect_column_mean_to_be_between(column=column_name, min_value=lower_limit, max_value=upper_limit)
    assert output["success"], f"Mean of column {column_name} is not within the range {lower_limit} - {upper_limit}"


@pytest.mark.parametrize("column_name, upper_limit", 
                         [("SepalLengthCm", 10), ("SepalWidthCm", 5), 
                          ("PetalLengthCm", 10), ("PetalWidthCm", 5)])
def test_column_max_range(df, column_name, upper_limit):
    # Check if the max of a column is within a certain range
    output = df.expect_column_max_to_be_between(column=column_name, max_value=upper_limit)
    assert output["success"], f"Max of column {column_name} is not within the range {upper_limit}"


@pytest.mark.parametrize("column_name, lower_limit", 
                         [("SepalLengthCm", 0.1), ("SepalWidthCm", 0.1), 
                          ("PetalLengthCm", 0.1), ("PetalWidthCm", 0.1)])
def test_column_min_range(df, column_name, lower_limit):
    # Check if the min of a column is within a certain range
    output = df.expect_column_min_to_be_between(column=column_name, min_value=lower_limit)
    assert output['success'], f"Min of column {column_name} is not within the range {lower_limit}"


@pytest.mark.parametrize("column_name, unique_values", [("Species", 3)])
def test_column_unique_values(df, column_name, unique_values):
    # Check if a column has a certain number of unique values
    output = df.expect_column_unique_value_count_to_be_between(column=column_name, min_value=unique_values, max_value=unique_values)
    assert output["success"], f"Column {column_name} does not have the correct number of unique values"


@pytest.mark.parametrize("column_name, unique_values", [("Species", ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])])
def test_column_unique_values_list(df, column_name, unique_values):
    # Check if a column has a certain number of unique values
    output = df.expect_column_unique_values_to_be_in_set(column=column_name, value_set=unique_values)
    assert output['success'], f"Column {column_name} does not have the correct number of unique values"


@pytest.mark.parametrize("columns", [['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']])
def test_if_columns_exist(df, columns):
    # Check if a column exists
    assert set(columns).issubset(df.columns), f"Columns {set(columns).difference(df.columns)} do not exist in DataFrame"


@pytest.mark.parametrize("column_name, ascending", [("Id", True)])
def test_if_df_is_sorted(df, column_name, ascending):
    # Check if a DataFrame is sorted
    assert df[column_name].is_monotonic_increasing == ascending, f"DataFrame is not sorted by {column_name}"


@pytest.mark.parametrize("column_name, unique_values", [("Species", ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])])
def test_column_unique_values_list(df, column_name, unique_values):
    # Check if a column has a certain number of unique values
    assert set(df[column_name].unique()) == set(unique_values), f"Column {column_name} does not have the correct number of unique values"
