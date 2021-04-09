"""
Path manipulation utilities.
"""

import os


def row2dest(row, root_dir=""):
    """
    Generate an output path for an image to download.

    Parameters
    ----------
    row : pandas.Series
        Entry from a SQUIDLE CSV file.
    root_dir : str, optional
        Top-level directory where images should be downloaded.
        Default is ``""``, an empty string.

    Returns
    -------
    destination : str
        Output path.
    """
    url = row["media_path"].strip()

    dirname = os.path.join(
        row["platform"],
        row["campaign_key"],
        row["deployment_key"],
    ).replace(" ", "_")
    fname = url.split("/")[-1]

    destination = os.path.join(root_dir, dirname, fname)
    return destination
