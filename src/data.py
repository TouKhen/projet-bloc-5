from pathlib import Path
import pandas as pd
import json

def load_file(path: str | Path) -> pd.DataFrame:
    """
    Load a JSON file and return its contents as a pandas DataFrame. The function expects
    the JSON file to be in a line-delimited format. If an error occurs during the
    reading process, a ValueError is raised with additional context about the failure.

    :param path: The file path to the JSON file. It can be provided as a string
        or a `Path` object.
    :type path: str | Path
    :return: A pandas DataFrame containing the data read from the JSON file. The
        DataFrame is populated with the parsed contents of the JSON file.
    :rtype: pd.DataFrame
    :raises ValueError: If reading the JSON file fails, an error message specifying
        the file path and the underlying error is raised.
    """
    try:
        return pd.read_json(path, lines=True)
    except Exception as e:
        raise ValueError(f"Failed to load JSON file from {path}: {str(e)}")



def load_multiple_files(paths: list[str]) -> pd.DataFrame:
    global_df = pd.DataFrame()

    for path in paths:
        df = load_file(path)
        global_df = pd.concat([global_df, df], ignore_index=True)

    return global_df