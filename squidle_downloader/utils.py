"""
Utility functions.
"""


def unique_map(arr):
    """
    Generate a mapping from unique value in an array to their locations.

    Parameters
    ----------
    arr : array_like
        Input array, containing repeated values.

    Returns
    -------
    mapping : dict
        Dictionary whose keys are the unique values in `arr`, and values are a
        list of indices to the entries in `arr` whose value is that key.
    """
    mapping = {}
    for i, x in enumerate(arr):
        if x not in mapping:
            mapping[x] = [i]
        else:
            mapping[x].append(i)
    return mapping


def clean_df(df, inplace=True):
    """
    Clean SQUIDLE dataframe.

    Removes trailing spaces from the ``"media_path"`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in SQUIDLE CSV format.
    inplace : bool, optional
        Whether the cleaning operation should be inplace.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataframe.
    """
    if not inplace:
        df = df.copy()
    df["media_path"] = df["media_path"].str.strip()
    return df
