#!/usr/bin/env python

"""
Module for downloading SQUIDLE database resources as CSV file(s).
"""

import datetime
import functools
import json
import os
import sys
import time

import pandas as pd
import requests
import tqdm

from . import __meta__, utils


def download_resource(resource, subdomain="", max_pages=None, verbose=1):
    """
    Download the entirety of a resource from SQUIDLE to a DataFrame.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    destination : str
        Name of output file.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    max_pages : int or None, optional
        Maximum number of pages to download.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    base_url = utils.build_base_url(subdomain)

    if verbose >= 1:
        print('Downloading all "{}" data from {}'.format(resource, base_url))

    url = base_url + "/api/" + resource
    params = {
        "q": json.dumps({"order_by": [{"field": "id", "direction": "asc"}]}),
        "results_per_page": 10_000,
    }

    result = requests.get(url, params=params).json()
    n_page = result["total_pages"]

    df = utils.clean_df(pd.json_normalize(result["objects"]))

    if n_page <= 1:
        return df

    if max_pages is not None and n_page > max_pages:
        if verbose >= 1:
            print(
                "Only downloading first {} of {} pages of results".format(
                    max_pages, n_page
                )
            )
        n_page = max_pages

    if verbose >= 1:
        _range = functools.partial(tqdm.trange, initial=1, total=n_page)
    else:
        _range = range

    for page in _range(2, n_page + 1):
        params["page"] = page
        result = requests.get(url, params=params).json()
        df = df.append(
            utils.clean_df(pd.json_normalize(result["objects"])), ignore_index=True
        )

    return df


def download_resource_to_csv(
    resource, destination, subdomain="", max_pages=None, verbose=1
):
    """
    Download the entirety of a resource from SQUIDLE to a CSV file.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    destination : str
        Name of output file.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    max_pages : int or None, optional
        Maximum number of pages to download.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    None
    """
    t0 = time.time()
    if verbose >= 1:
        print(
            'Will downloading "{}" data to CSV file {}'.format(
                resource,
                destination,
            )
        )
    df = download_resource(
        resource, subdomain=subdomain, max_pages=max_pages, verbose=verbose
    )
    if verbose >= 1:
        print("Saving output dataset to {}".format(destination))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    df.to_csv(destination, index=False)
    if verbose >= 1:
        print(
            "Downloaded {} {} records in {}".format(
                len(df),
                resource,
                datetime.timedelta(seconds=time.time() - t0),
            )
        )


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
        "resource",
        type=str,
        help="Name of API resource (platform, campaign, deployment, media, pose, etc)",
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--subdomain",
        type=str,
        default="",
        help='SQUIDLE subdomain access (e.g. "soi").',
    )
    parser.add_argument(
        "--pages",
        dest="max_pages",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Maximum number of pages to download. By default, all pages"
            " are downloaded."
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
    Run command line interface for downloading a SQUIDLE resource as a CSV file.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_resource_to_csv(**kwargs)


if __name__ == "__main__":
    main()
