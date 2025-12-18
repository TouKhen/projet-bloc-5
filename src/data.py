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


def load_large_jsonl(filepath, chunksize=10000):
    """
    Loads large JSON Lines (JSONL) file into a pandas DataFrame in chunks.

    This function is designed to read a JSONL file in manageable chunks for files
    that are too large to be loaded into memory at once. It processes the file in
    segments and combines the chunks into a single DataFrame.

    :param filepath: The path to the JSONL file to be loaded.
    :param chunksize: The number of lines to be read in each chunk from the JSONL
        file. Defaults to 10,000 lines.
    :return: A pandas DataFrame containing the concatenated data from all the
        chunks.
    """
    chunks = []
    reader = pd.read_json(filepath, lines=True, chunksize=chunksize)

    for chunk in reader:
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_jsonl_part(
    path: str | Path,
    *,
    start: int = 0,
    nrows: int | None = None,
    columns: list[str] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Load only a part of a JSON Lines (JSONL) file into a DataFrame.

    This is useful for very large files when you only want:
    - a slice of rows (via `start` + `nrows`), and/or
    - a subset of columns (via `columns`)

    Notes:
    - This function assumes JSONL (one JSON object per line). For a single large
      JSON array/object file, this chunked approach won't work reliably.
    - Row indexing is 0-based.

    :param path: Path to the JSONL file.
    :param start: Number of rows to skip before starting to collect data.
    :param nrows: Number of rows to return. If None, returns everything after `start`.
    :param columns: Optional list of column names to keep (others are dropped).
    :param chunksize: Number of rows per chunk to read from disk.
    :return: A DataFrame containing the requested slice.
    :raises ValueError: If parameters are invalid or reading fails.
    """
    if start < 0:
        raise ValueError("start must be >= 0")
    if nrows is not None and nrows < 0:
        raise ValueError("nrows must be >= 0 or None")
    if nrows == 0:
        return pd.DataFrame(columns=columns if columns is not None else None)

    remaining_to_skip = start
    remaining_to_take = nrows
    out: list[pd.DataFrame] = []

    try:
        reader = pd.read_json(path, lines=True, chunksize=chunksize)

        for chunk in reader:
            # Skip whole chunks if needed
            if remaining_to_skip > 0:
                if remaining_to_skip >= len(chunk):
                    remaining_to_skip -= len(chunk)
                    continue
                chunk = chunk.iloc[remaining_to_skip:]
                remaining_to_skip = 0

            # Take only what we need
            if remaining_to_take is not None:
                if remaining_to_take <= 0:
                    break
                if len(chunk) > remaining_to_take:
                    chunk = chunk.iloc[:remaining_to_take]
                remaining_to_take -= len(chunk)

            if columns is not None:
                # Keep only requested columns that exist in this chunk
                keep = [c for c in columns if c in chunk.columns]
                chunk = chunk.loc[:, keep]

            out.append(chunk)

            if remaining_to_take is not None and remaining_to_take <= 0:
                break

        if not out:
            return pd.DataFrame(columns=columns if columns is not None else None)

        return pd.concat(out, ignore_index=True)

    except Exception as e:
        raise ValueError(f"Failed to load part of JSONL file from {path}: {str(e)}")


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
        df = load_large_jsonl(path)
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
    if isinstance(path, Path):
        path = str(path)

        # Process in chunks to manage memory efficiently
    chunk_size = 50000  # Adjust based on your system's available memory

    with open(path, 'w', encoding='utf-8') as f:
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]

            # Use to_json with lines=True on smaller chunks
            chunk_json = chunk.to_json(orient='records', lines=True)
            f.write(chunk_json)

            # Add newline if not the last chunk and chunk doesn't end with newline
            if end_idx < len(df) and not chunk_json.endswith('\n'):
                f.write('\n')



def clean_items_data(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses a DataFrame containing item data. This includes steps such as
    removing unnecessary columns, handling duplicates, filling missing data, and converting
    price information to a numeric format.

    :param items_df: DataFrame containing item data to be cleaned and preprocessed.
                     It is expected to contain the following columns:
                     'images', 'videos', 'bought_together', 'parent_asin', 'price', and 'main_category'.
    :type items_df: pd.DataFrame
    :return: A cleaned and preprocessed DataFrame with the following transformations:
             - Columns 'images', 'videos', and 'bought_together' are removed.
             - Duplicate rows based on 'parent_asin' are dropped, keeping the first occurrence.
             - Rows with missing or empty 'parent_asin' are removed.
             - The 'price' column is converted to numeric and invalid entries coerced into NaN.
             - Missing values in the 'price' column are filled with the mean price of their
               respective 'main_category'.
    :rtype: pd.DataFrame
    """
    items_cleaned = items_df.copy()

    # Remove images and videos columns
    items_cleaned = items_cleaned.drop(columns=['images', 'videos', 'bought_together'])

    # Remove duplicate rows based on parent_asin
    items_cleaned = items_cleaned.drop_duplicates(subset=['parent_asin'], keep='first')

    # Remove rows with missing parent_asin
    items_cleaned = items_cleaned[items_cleaned['parent_asin'].notna()]
    items_cleaned = items_cleaned[items_cleaned['parent_asin'] != '']

    # Convert price to numeric
    items_cleaned['price'] = pd.to_numeric(items_df['price'], errors='coerce')
    
    # Fill NaN prices with the mean price of main_category
    items_cleaned['price'] = items_cleaned.groupby('main_category')['price'].transform(lambda x: x.fillna(round(x.mean(), 2)))

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

    reviews_cleaned = reviews_cleaned.drop(columns=['title', 'text', 'images'])

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


def cross_validation(models, cv=5):
    """
    Performs k-fold cross-validation for a given set of models and computes evaluation
    metrics including accuracy, precision, recall, and F1 score based on the provided
    data and specified number of splits. The function returns a dictionary containing
    the mean and standard deviation for each computed metric.

    Key metrics evaluated:
    - Accuracy
    - Precision
    - Recall
    - F1 score

    :param models: A dictionary where each key is the model name (str) and the value
        is another dictionary containing the model ('model') and its corresponding
        cleaned data ('data'). The 'data' object should have `X_clean` and `y_clean`
        attributes representing features and labels respectively.
    :param cv: An integer specifying the number of folds for k-fold cross-validation.
        The default value is 5.

    :return: A dictionary where each key corresponds to a model name and contains
        nested evaluation metrics. Each metric has its mean and standard deviation.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
        recall_score, f1_score

    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    # Create KFold cross-validator
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Dictionary to store results
    cv_results = {}

    # Perform cross-validation for each model
    for name, model in models.items():
        cv_results[name] = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model['model'], model['data'].X_clean, model['data'].y_clean, cv=kf, scoring=scorer)
            cv_results[name][metric_name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }

    return cv_results