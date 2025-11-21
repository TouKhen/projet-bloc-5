from pathlib import Path
import pandas as pd

def load_file(path: str | Path) -> pd.DataFrame:
    """
    Loads a JSON file from the specified path and returns its content as a pandas DataFrame.
    The JSON file is expected to be in a line-delimited format.

    :param path: Path to the JSON file. It can be a string or a pathlib.Path object.
    :type path: str | Path
    :return: Content of the JSON file as a pandas DataFrame.
    :rtype: pd.DataFrame
    :raises ValueError: If the file cannot be loaded or if there is an issue while
                        reading the JSON file.
    """
    try:
        return pd.read_json(path, lines=True)
    except Exception as e:
        raise ValueError(f"Failed to load JSON file from {path}: {str(e)}")



def load_multiple_files(paths: list[str]) -> pd.DataFrame:
    """
    Load data from multiple files and concatenate them into a single DataFrame.

    This function takes a list of file paths, loads the content of each file
    into a DataFrame, and combines all these DataFrames into a single
    DataFrame using an index-ignoring concatenation operation. It is useful
    when working with datasets split across multiple files, ensuring they
    can be processed as a whole.

    :param paths:
        A list of file paths to load data from.
    :return:
        A concatenated DataFrame containing data from all the provided files.
    """
    global_df = pd.DataFrame()

    for path in paths:
        df = load_file(path)
        global_df = pd.concat([global_df, df], ignore_index=True)

    return global_df


def save_file(df: pd.DataFrame, path: str | Path) -> None:
    """
    Saves a DataFrame to a file in JSON Lines format. This function is used to serialize
    the data within the DataFrame into a JSON file at the specified path. The output JSON
    file will store each row of the DataFrame as a separate JSON object.

    :param df: The DataFrame to be saved. It contains data to be serialized into JSON Lines format.
    :type df: pd.DataFrame
    :param path: The file path where the JSON Lines file will be saved. It can be a string or a Path object.
    :type path: str | Path
    :return: None
    """
    df.to_json(path, orient="records", lines=True)


def clean_items_data(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes a DataFrame containing item data for further analysis or processing.
    The function removes unnecessary columns (images, videos), duplicate rows based on
    the "parent_asin" column, and rows with missing or invalid "parent_asin" values. It also
    converts the "price" column to a numeric data type.

    :param items_df: A pandas DataFrame containing item data that needs to be cleaned. The
        DataFrame must include columns 'images', 'videos', 'parent_asin', and 'price'.
    :type items_df: pd.DataFrame
    :return: A cleaned pandas DataFrame with unnecessary columns removed, duplicates handled,
        invalid rows filtered, and the "price" column converted to numeric format.
    :rtype: pd.DataFrame
    """
    items_cleaned = items_df.copy()

    # Remove images and videos columns
    items_cleaned = items_cleaned.drop(columns=['images', 'videos'])

    # Remove duplicate rows based on parent_asin
    items_cleaned = items_cleaned.drop_duplicates(subset=['parent_asin'], keep='first')

    # Remove rows with missing parent_asin
    items_cleaned = items_cleaned[items_cleaned['parent_asin'].notna()]
    items_cleaned = items_cleaned[items_cleaned['parent_asin'] != '']

    # Convert price to numeric
    items_cleaned['price'] = pd.to_numeric(items_df['price'], errors='coerce')

    return items_cleaned


def clean_reviews_data(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes a reviews DataFrame by performing the following tasks:
    - Converts non-hashable list-like columns into strings to ensure compatibility with drop_duplicates().
    - Removes duplicate rows from the DataFrame.
    - Eliminates rows where critical fields such as 'user_id' or 'parent_asin' are missing.
    - Ensures that the 'rating' column is properly converted to numeric values.

    :param reviews_df: The input DataFrame containing review details. Expected to have columns such as
        'user_id', 'parent_asin', 'rating', and others. List-like data within object-typed columns will
        be converted to strings during processing.
    :type reviews_df: pandas.DataFrame
    :return: A cleaned DataFrame with duplicate rows removed, critical missing values handled, and
        proper numeric type for the 'rating' field.
    :rtype: pandas.DataFrame
    """
    reviews_cleaned = reviews_df.copy()

    # Convert list columns to strings to make them hashable
    # This is necessary for drop_duplicates() to work
    for col in reviews_cleaned.columns:
        if reviews_cleaned[col].dtype == 'object':
            # Check if the column contains lists
            sample = reviews_cleaned[col].dropna().iloc[0] if len(reviews_cleaned[col].dropna()) > 0 else None
            if isinstance(sample, list):
                # Convert lists to strings (comma-separated or JSON format)
                reviews_cleaned[col] = reviews_cleaned[col].apply(
                    lambda x: ','.join(map(str, x)) if isinstance(x, list) else x
                )

    # Remove duplicate rows
    reviews_cleaned = reviews_cleaned.drop_duplicates()

    # Remove rows with missing user_id or parent_asin
    reviews_cleaned = reviews_cleaned[reviews_cleaned['user_id'].notna()]
    reviews_cleaned = reviews_cleaned[reviews_cleaned['parent_asin'].notna()]

    # Convert rating to numeric
    reviews_cleaned['rating'] = pd.to_numeric(reviews_cleaned['rating'], errors='coerce')

    return reviews_cleaned
