#!/usr/bin/env python

"""
Module for building a dataset of unlabelled SQUIDLE images.
"""

import datetime
import os
import sys
import time

import pandas as pd

from . import __meta__, download_resource, utils


def build_dataset(cache_dir, subdomain="", max_pages=None, force=False, verbose=1):
    """
    Build an unlabelled dataset of images on a SQUIDLE subdomain.

    Parameters
    ----------
    cache_dir : str
        Path to cache directory.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    max_pages : int or None, optional
        Maximum number of pages to download. If `None` (default), all pages are
        downloaded.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    if verbose >= 1:
        print(
            "Building unlabelled image dataset from {}".format(
                utils.build_base_url(subdomain),
            ),
            flush=True,
        )
    df_media = download_resource.cached_download(
        "media",
        cache_dir,
        subdomain=subdomain,
        max_pages=max_pages,
        force=force,
        verbose=verbose - 1,
    )
    df_pose = download_resource.cached_download(
        "pose",
        cache_dir,
        subdomain=subdomain,
        max_pages=max_pages,
        force=force,
        verbose=verbose - 1,
    )
    df_deployment = download_resource.cached_download(
        "deployment",
        cache_dir,
        subdomain=subdomain,
        force=force,
        verbose=verbose - 1,
    )

    if verbose >= 2:
        print(
            "Joining {} resources into a single dataframe".format(
                utils.build_base_url(subdomain),
            ),
            flush=True,
        )

    # Drop invalid records
    df_media = df_media[df_media["is_valid"]]
    # Drop records which are not images
    df_media = df_media[df_media["media_type.name"] == "image"]

    # Drop superfluous columns by selecting only the columns we care about
    # and rename columns as needed for merger
    df_media = df_media[["id", "key", "path_best", "timestamp_start", "deployment_id"]]

    if "media.id" not in df_pose.columns:
        # soi subdomain
        df_pose.rename(columns={"id": "media.id"}, inplace=True)
    df_pose = df_pose[["alt", "dep", "lat", "lon", "media.id"]]

    if "campaign.id" not in df_pose.columns:
        df_deployment.rename(columns={"campaign_id": "campaign.id"}, inplace=True)
    df_deployment.drop(
        columns=["campaign.id", "color", "created_at", "is_valid", "platform.id"],
        inplace=True,
        errors="ignore",
    )
    df_deployment.rename(
        columns={
            "name": "deployment",
            "key": "deployment_key",
            "id": "deployment_id",
        },
        inplace=True,
    )

    # Merge pose data into media data to get latitude, longitude, depth
    df = pd.merge(
        df_media, df_pose, left_on="id", right_on="media.id", how="left", copy=False
    )
    df.drop(columns="media.id", inplace=True)

    # Merge deployment data to get deployment, campaign and platform names
    df = df.merge(df_deployment, on="deployment_id", how="left", copy=False)
    df.drop(columns=["deployment_id"], inplace=True)

    # Rename columns
    df.rename(
        columns={
            "path_best": "url",
            "timestamp_start": "timestamp",
            "alt": "altitude",
            "dep": "depth",
            "lat": "latitude",
            "lon": "longitude",
            "campaign.name": "campaign",
            "platform.name": "platform",
        },
        inplace=True,
    )

    # Remove entries without a url
    df = df[df["url"].notna()]
    df = df[df["url"] != ""]

    # Remove trailing whitespace from string fields
    df = utils.clean_df(df)

    return df


def build_dataset_multidomain(cache_dir, subdomains=("", "soi"), verbose=1, **kwargs):
    """
    Build an unlabelled dataset of images on a SQUIDLE subdomain.

    Parameters
    ----------
    cache_dir : str
        Path to cache directory.
    subdomains : iterable, optional
        SQUIDLE subdomains to use. Default is `("", "soi")`.
    max_pages : int or None, optional
        Maximum number of pages to download for each subdomain. If `None`
        (default), all pages are downloaded.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    if verbose >= 1:
        print(
            "Building unlabelled SQUIDLE dataset for {} domains".format(
                len(subdomains),
            ),
            flush=True,
        )
    stacked_df = None
    for i_subdomain, subdomain in enumerate(subdomains):
        if verbose >= 2 and len(subdomains) > 1:
            print(
                "  Subdomain {}/{}: {}".format(
                    i_subdomain + 1, len(subdomains), utils.build_base_url(subdomain)
                )
            )
        df_i = build_dataset(
            cache_dir,
            subdomain=subdomain,
            **kwargs,
            verbose=verbose,
        )
        if stacked_df is None:
            stacked_df = df_i
        else:
            stacked_df = stacked_df.append(df_i, ignore_index=True)

    stacked_df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    stacked_df.drop(columns=["id"], inplace=True)

    return stacked_df


def dataset_multidomain_to_csv(destination, *args, verbose=1, **kwargs):
    """
    Build an unlabelled dataset of images on a SQUIDLE subdomain.

    Parameters
    ----------
    destination : str
        Name of output file.
    cache_dir : str
        Path to cache directory.
    subdomains : iterable, optional
        SQUIDLE subdomains to use. Default is `("", "soi")`.
    max_pages : int or None, optional
        Maximum number of pages to download for each subdomain. If `None`
        (default), all pages are downloaded.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    t0 = time.time()

    if verbose >= 1:
        print(
            "Will build unlabelled SQUIDLE dataset and save to CSV file {}".format(
                destination,
            ),
            flush=True,
        )

    df = build_dataset_multidomain(*args, **kwargs, verbose=verbose)

    if verbose >= 1:
        print("Saving output dataset to {}".format(destination), flush=True)
    destdir = os.path.dirname(destination)
    if destdir and not os.path.isdir(destdir):
        os.makedirs(destdir, exist_ok=True)
    df.to_csv(destination, index=False)
    if verbose >= 1:
        print(
            "Built dataset containing {} unlabelled images in {}".format(
                len(df),
                datetime.timedelta(seconds=time.time() - t0),
            ),
            flush=True,
        )

    return df


def get_parser():
    """
    Build CLI parser for downloading SQUIDLE resources.

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
        description="Download a resource from SQUIDLE to CSV",
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
        "destination",
        type=str,
        help="Output CSV file.",
    )
    parser.add_argument(
        "cache_dir",
        metavar="cache",
        type=str,
        help="Path to cache directory.",
    )
    parser.add_argument(
        "--subdomains",
        metavar="SUBDOMAIN",
        nargs="*",
        type=str,
        default=("", "soi"),
        help="SQUIDLE subdomain access. Default: %(default)s.",
    )
    parser.add_argument(
        "--pages",
        dest="max_pages",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Maximum number of pages of media files to utilise from each"
            " subdomain. Each page contains ten thousand records. By default,"
            " all pages are downloaded."
        ),
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
    dataset_multidomain_to_csv(**kwargs)


if __name__ == "__main__":
    main()
