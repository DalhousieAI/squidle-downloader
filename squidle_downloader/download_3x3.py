#!/usr/bin/env python

"""
Module for downloading a dataset with labels specified as a 3x3 grid.

Used to download the data contained in:

    Yamada, Prügel‐Bennett, Thornton. Learning features from georeferenced
    seafloor imagery with location guided autoencoders. J Field Robotics.
    2021; 38: 52–67. `doi:10.1002/rob.21961 <https://doi.org/10.1002/rob.21961>`_
"""

import datetime
import math
import os
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import PIL.Image
from tqdm.autonotebook import tqdm

from . import __meta__, utils
from .path import row2dest


def crop_ninth(img, x, y):
    """
    Crop an image down to a patch a third of the width and height.

    Parameters
    ----------
    img : PIL.Image
        Image object.
    x : float
        Center x coordinate for the crop patch, specified as a fraction of the
        image width ``0 < x < 1``.
    y : float
        Center y coordinate for the crop patch, specified as a fraction of the
        image height ``0 < y < 1``.

    Returns
    -------
    subimg : PIL.Image
        Cropped image.
    """
    w = math.ceil(img.size[0] / 3)
    h = math.ceil(img.size[1] / 3)
    lft = max(0, math.floor((x - 1 / 6) * img.size[0]))
    top = max(0, math.floor((y - 1 / 6) * img.size[1]))
    rgt = min(img.size[0], lft + w)
    btm = min(img.size[1], top + h)
    return img.crop((lft, top, rgt, btm))


def download_dataset_3x3(df, data_dir, verbose=1):
    """
    Download a SQUIDLE dataset with images labelled 3x3.

    Parameters
    ----------
    df : Pandas.DataFrame
        Input data, as loaded from a SQUIDLE CSV file.
    data_dir : str
        Path to root directory to contain the cropped downloaded images.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset, like `df`, but with a ``"path"`` column containing the
        path to the output files.
    """
    df = utils.clean_df(df)
    output_df = pd.DataFrame()

    url2idx = utils.unique_map(df["media_path"])

    if verbose >= 1:
        print("Processing {} labels over {} images".format(len(df), len(url2idx)))

    skipped_points = defaultdict(lambda: 0)
    n_already_downloaded = 0
    n_download = 0

    for i_url, (url, indices) in enumerate(tqdm(url2idx.items(), disable=verbose != 1)):

        if verbose >= 2:
            print("{:4d}/{:4d}: {}".format(i_url, len(url2idx), url))

        todo = []
        any_missing = False
        for idx in indices:
            row = df.iloc[idx]
            if row["media_path"] != url:
                raise ValueError(
                    "Mismatched url:\n{}\nvs\n{}".format(url, row["media_path"])
                )

            x_coord = (6 * row["x"] - 1) / 2
            y_coord = (6 * row["y"] - 1) / 2
            eps = 1e-4
            if (
                np.abs(x_coord - np.round(x_coord)) > eps
                or np.abs(y_coord - np.round(y_coord)) > eps
            ):
                print(
                    "Warning:\n"
                    "  While handling image {}\n"
                    "  Point {}, {} is not a 3x3 grid center".format(
                        url, row["x"], row["y"]
                    )
                )
                skipped_points[url] += 1
                continue

            destination = row2dest(row)
            base, ext = os.path.splitext(destination)
            rel_dest = "{}_c{}-{}{}".format(
                base, int(np.round(x_coord)), int(np.round(y_coord)), ext
            )
            if not os.path.isfile(os.path.join(data_dir, rel_dest)):
                any_missing = True
            todo.append((row, rel_dest))

        img = None
        if any_missing:
            if verbose >= 3:
                print("    Downloading media url to temporary file")
            with tempfile.NamedTemporaryFile() as f:
                _, headers = urllib.request.urlretrieve(url.strip(), f.name)
                img = PIL.Image.open(f.name)
            n_download += 1
        elif verbose >= 3:
            n_already_downloaded += 1
            print("    All output files already exist; skipping download")

        for i_todo, (row, rel_dest) in enumerate(todo):
            if img is not None:
                subimg = crop_ninth(img, row["x"], row["y"])
                if verbose >= 4:
                    print(
                        "    {}/{} Saving ({:.4f}, {:.4f}) to {}".format(
                            i_todo,
                            len(todo),
                            row["x"],
                            row["y"],
                            os.path.join(data_dir, destination),
                        )
                    )
                destination = os.path.join(data_dir, rel_dest)
                destdir = os.path.dirname(destination)
                if destdir and not os.path.isdir(destdir):
                    os.makedirs(destdir, exist_ok=True)
                subimg.save(os.path.join(data_dir, destination))
            row = row.copy()
            row["path"] = destination
            output_df = output_df.append(row)

    if verbose >= 1:
        print(
            "Finished processing {} labels across {} images".format(
                len(df) - len(skipped_points), len(url2idx)
            )
        )
        print(
            "There were {} images already downloaded. The remaining {} images"
            " were downloaded.".format(n_already_downloaded, n_download)
        )
        if len(skipped_points) > 0:
            print("Skipped {} labels which were off-grid.".format(len(skipped_points)))
    return output_df


def download_from_csv(input_csv, output_csv, data_dir, verbose=1):
    """
    Download a SQUIDLE dataset with images labelled 3x3 from CSV file.

    Parameters
    ----------
    input_csv : str
        Path to input CSV file.
    output_csv : str
        Path to output CSV file.
    data_dir : str
        Path to root directory to contain the cropped downloaded images.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    None
    """
    t0 = time.time()
    if verbose >= 1:
        print("Processing file: {}".format(input_csv))
    df = pd.read_csv(input_csv)
    if verbose >= 1:
        print("  containing {} records".format(len(df)))
        print("Cropped images will be saved in: {}".format(data_dir))
        print("Output dataset will be saved to: {}".format(output_csv))
    output_df = download_dataset_3x3(df, data_dir=data_dir, verbose=verbose)
    if verbose >= 1:
        print("Saving output dataset to {}".format(output_csv))
    output_df.to_csv(output_csv, index=False)
    print("Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)))


def get_parser():
    """
    Build CLI parser for downloading a 3x3 image dataset.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download a dataset labelled 3x3, from a SQUIDLE CSV file",
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=__meta__.version),
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file, in the format generated by SQUIDLE.",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Output CSV file.",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root directory for downloaded images.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="""
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is %(default)s.
        """,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="""
            Decrease the level of verbosity of the program. This can be
            specified multiple times, each will reduce the amount of detail
            printed to the terminal.
        """,
    )
    return parser


def main():
    """
    Run command line interface for downloading 3x3 labelled images.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_from_csv(**kwargs)


if __name__ == "__main__":
    main()
