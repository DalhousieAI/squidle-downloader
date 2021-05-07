"""
Utility functions.
"""


def remove_prefix(text, prefix):
    """
    Remove a prefix from a string.

    Parameters
    ----------
    text : str
        String possibly starting with `prefix`.
    prefix : str
        The prefix string to remove.

    Returns
    -------
    str
        String like `text`, but with `prefix` removed if it occured at the
        start of `text`.
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


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
    if "media_path" in df.columns:
        df["media_path"] = df["media_path"].str.strip()
    return df


def build_base_url(subdomain):
    """
    Construct SQUIDLE base url, with subdomain included if necessary.

    Parameters
    ----------
    subdomain : str
        SQUIDLE subdomain to use, one of `""` or `"soi"`.

    Returns
    -------
    url : str
        SQUIDLE url with subdomain included.
    """
    if subdomain:
        subdomain += "."
    return "https://{}squidle.org".format(subdomain)


def count_lines(filename):
    """
    Count the number of lines in a file.

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    -------
    int
        Number of lines in file.
    """
    with open(filename, "rb") as f:
        for _i, _ in enumerate(f):
            pass
    return _i + 1
