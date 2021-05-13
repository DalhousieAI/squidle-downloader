#!/usr/bin/env python

"""
Downloading images listed in a SQUIDLE formatted CSV file.
"""

import datetime
import functools
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import tqdm

from . import __meta__, utils


def download_images_from_dataframe(
    df, output_dir, skip_existing=True, delete_partial=True, verbose=1, use_tqdm=True
):
    """
    Download all images from a dataframe in SQUIDLE format.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of images to download.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is `True`.
    delete_partial : bool, optional
        Whether to delete partially downloaded files in the event of an error,
        such as running out of disk space or keyboard interrupt.
        Default is `True`.
    verbose : int, optional
        Verbosity level. Default is `1`.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if writing to a file.
        Default is `True`.

    Returns
    -------
    None
    """
    if verbose >= 1:
        print("Downloading {} images".format(len(df)), flush=True)

    if verbose == 1 and use_tqdm:
        maybe_tqdm = functools.partial(tqdm.tqdm, total=len(df))
    else:
        maybe_tqdm = lambda x: x  # noqa: E731

    n_already_downloaded = 0
    n_download = 0

    t0 = time.time()

    for i_row, row in maybe_tqdm(df.iterrows()):
        ext = os.path.splitext(row["url"])[-1]
        destination = os.path.join(
            output_dir,
            row["campaign"],
            row["deployment"],
            row["key"] + ext,
        )
        if (
            verbose >= 1
            and not use_tqdm
            and i_row > 0
            and (i_row <= 5 or i_row % 100 == 0)
        ):
            t_elapsed = time.time() - t0
            t_remain = t_elapsed / i_row * (len(df) - i_row)
            print(
                "Processed {:4d}/{} urls ({:6.2f}%) in {} (approx. {} remaining)"
                "".format(
                    i_row,
                    len(df),
                    100 * i_row / len(df),
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                ),
                flush=True,
            )

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if skip_existing and os.path.isfile(destination):
            n_already_downloaded += 1
            if verbose >= 3:
                print(
                    "    Skipping download of {}\n"
                    "    Destination exists: {}".format(row["url"], destination),
                    flush=True,
                )
            continue
        if verbose >= 2:
            print(
                "    Downloading {} to {}".format(row["url"], destination), flush=True
            )
        try:
            _, headers = urllib.request.urlretrieve(row["url"], filename=destination)
            n_download += 1
        except BaseException:
            if os.path.isfile(destination) and delete_partial:
                print(
                    "An error occured while processing {}.\n"
                    "Deleting partial file {}".format(row["url"], destination)
                )
                os.remove(destination)
            raise

    if verbose >= 1:
        print("Finished processing {} images".format(len(df)))
        print(
            "There were {} images already downloaded. The remaining {} images"
            " were downloaded.".format(n_already_downloaded, n_download),
            flush=True,
        )
    return


def download_images_from_csv(
    input_csv,
    output_dir,
    *args,
    skip_existing=True,
    verbose=1,
    i_part=None,
    n_part=None,
    **kwargs
):
    """
    Download all images from a CSV file with contents in SQUIDLE format.

    Parameters
    ----------
    input_csv : str
        Path to CSV file.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is `True`.
    verbose : int, optional
        Verbosity level. Default is `1`.
    i_part : int or None, optional
        Run on only a partition of the CSV file. If `None` (default), the
        entire dataset will be downloaded by this process. Otherwise, `n_part`
        must also be set.
    n_part : int or None, optional
        Number of partitions being run. Default is `None`.
    **kwargs : optional
        Additional arguments as per `download_images_from_dataframe`.

    Returns
    -------
    None
    """
    t0 = time.time()
    if (i_part is not None and n_part is None) or (
        i_part is None and n_part is not None
    ):
        raise ValueError(
            "Both `i_part` and `n_part` must be defined when partitioning"
            " the CSV file."
        )
    skiprows = []
    if n_part:
        part_str = "(part {} of {})".format(i_part, n_part)
        n_lines = utils.count_lines(input_csv) - 1
        partition_size = n_lines / n_part
        i_part == 0 if i_part == n_part else i_part
        start_idx = round(i_part * partition_size)
        end_idx = round((i_part + 1) * partition_size)
        skiprows = list(range(1, 1 + start_idx)) + list(range(1 + end_idx, 1 + n_lines))

    if verbose >= 1:
        print(
            "Will download {} images {}listed in {}".format(
                "all" if not n_part else end_idx - start_idx,
                "" if not n_part else part_str + " ",
                input_csv,
            )
        )
        print("To output directory {}".format(output_dir))
        if skip_existing:
            print("Existing outputs will be skipped.", flush=True)
        else:
            print("Existing outputs will be overwritten.", flush=True)
        print("Reading CSV file...")
    df = pd.read_csv(
        input_csv,
        dtype={
            "key": str,
            "url": str,
            "timestamp": str,
            "altitude": np.float32,
            "depth": np.float32,
            "latitude": np.float32,
            "longitude": np.float32,
            "deployment_key": str,
            "deployment": str,
            "campaign": str,
            "platform": str,
        },
        parse_dates=["timestamp"],
        skiprows=skiprows,
    )
    if verbose >= 1:
        print("Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True)
    ret = download_images_from_dataframe(
        df, output_dir, *args, skip_existing=skip_existing, verbose=verbose, **kwargs
    )
    print(
        "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
        flush=True,
    )
    return ret


def get_parser():
    """
    Build CLI parser for downloading SQUIDLE image dataset.

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
        description="Download all images listed in a SQUIDLE CSV file",
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
        "output_dir",
        type=str,
        help="Root directory for downloaded images.",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--npart",
        dest="n_part",
        type=int,
        help="Number of processing partitions being run.",
    )
    parser.add_argument(
        "--part",
        "--ipart",
        dest="i_part",
        type=int,
        help="Partition index for this process.",
    )
    parser.add_argument(
        "--clobber",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing outputs instead of skipping their download.",
    )
    parser.add_argument(
        "--keep-on-error",
        dest="delete_partial",
        action="store_false",
        help="Keep partially downloaded files in the event of an error.",
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
    Run command line interface for downloading images.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_images_from_csv(**kwargs)


if __name__ == "__main__":
    main()
