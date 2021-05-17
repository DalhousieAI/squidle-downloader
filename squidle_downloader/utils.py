"""
Utility functions.
"""

import os


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


def file_size(filename, sf=3):
    """
    Get the size of a file, in human readable units.

    Parameters
    ----------
    filename : str
        Path to file.
    sf : int, optional
        Number of significant figures to include. Default is `3`.

    Returns
    -------
    str
        The size of the file in human readable format.
    """

    def convert_bytes(num):
        """
        Convert units of bytes into human readable kB, MB, GB, or TB.
        """
        for unit in ["bytes", "kB", "MB", "GB", "TB"]:
            if num >= 1024:
                num /= 1024
                continue
            digits = sf
            if num >= 1:
                digits -= 1
            if num >= 10:
                digits -= 1
            if num >= 100:
                digits -= 1
            if num >= 1000:
                digits -= 1
            digits = max(0, digits)
            fmt = "{:." + str(digits) + "f} {}"
            return fmt.format(num, unit)

    return convert_bytes(os.stat(filename).st_size)


def sanitize_filename_series(series, allow_dotfiles=False):
    """
    Sanitise strings in a Series, to remove characters forbidden in filenames.

    Parameters
    ----------
    series : pandas.Series
        Input Series of values.
    allow_dotfiles : bool, optional
        Whether to allow leading periods. Leading periods indicate hidden files
        on Linux. Default is `False`.

    Returns
    -------
    pandas.Series
        Sanitised version of `series`.
    """
    # Folder names cannot end with a period in Windows. In Unix, a leading
    # period means the file or folder is normally hidden.
    # For this reason, we trim away leading and trailing periods as well as
    # spaces.
    if allow_dotfiles:
        series = series.str.strip()
        series = series.str.rstrip(".")
    else:
        series = series.str.strip(" .")

    # On Windows, a filename cannot contain any of the following characters:
    # \ / : * ? " < > |
    # Other operating systems are more permissive.
    # Replace / with a hyphen
    series = series.str.replace("/", "-")
    # Use a blacklist to remove any remaining forbidden characters
    series = series.str.replace(r'[\/:*?"<>|]+', "")
    return series
