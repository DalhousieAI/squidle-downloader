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
    pandas.DataFrame
        Output dataset.
    """
    t0 = time.time()
    if verbose >= 1:
        print(
            'Will download "{}" data to CSV file {}'.format(
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
    return df


def get_cache_path(resource, root_cache_dir, subdomain="", page=None):
    """
    Build a path for caching an item.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    root_cache_dir : str
        Path to top level cache directory.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    page : int or None, optional
        Page number for paginated resources. Set to `None` (default) if the
        cache is not paginated.

    Returns
    -------
    cache_path : str
        Path for caching this (page of a) resource.
    """
    base = os.path.join(
        root_cache_dir,
        "root" if subdomain == "" else subdomain,
    )
    if page is None:
        inner = "{}.csv".format(resource)
    else:
        inner = os.path.join(resource, "{}_p{}.csv".format(resource, page))
    return os.path.join(base, inner)


def cached_download_nopagination(
    resource, cache_dir, subdomain="", force=False, return_updated=False, verbose=1
):
    """
    Download the entirety of a resource from SQUIDLE, caching the output.

    The results are all downloaded and cached in one file. There is no
    intermediate caching for different pages in the resource.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    cache_dir : str
        Path to cache directory.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    return_updated : bool, optional
        Whether to return whether the cache was updated. Default is `False`.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    df : pandas.DataFrame
        Output dataset.
    updated : bool, optional
        Whether the cache was updated.
    """
    fname = get_cache_path(resource, cache_dir, subdomain=subdomain)
    if os.path.isfile(fname) and not force:
        if verbose >= 1:
            print("Loading from cache {}".format(fname))
        df = pd.read_csv(fname)
        updated = False
    else:
        df = download_resource_to_csv(
            resource, fname, subdomain=subdomain, verbose=verbose
        )
        updated = True
    if return_updated:
        return df, updated
    else:
        return df


def cached_download_paginated(
    resource,
    cache_dir,
    subdomain="",
    force=False,
    max_pages=None,
    return_updated=False,
    verbose=1,
):
    """
    Download the entirety of a resource from SQUIDLE, caching each result page.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    cache_dir : str
        Path to cache directory.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    max_pages : int or None, optional
        Maximum number of pages to download. If `None` (default), all pages are
        downloaded.
    return_updated : bool, optional
        Whether to return whether the cache was updated. Default is `False`.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    df : pandas.DataFrame
        Output dataset.
    updated : bool, optional
        Whether the cache was updated.
    """
    base_url = utils.build_base_url(subdomain)
    url = base_url + "/api/" + resource
    params = {
        "q": json.dumps({"order_by": [{"field": "id", "direction": "asc"}]}),
        "results_per_page": 10_000,
    }

    if verbose >= 1:
        print('Downloading paginated "{}" data from {}'.format(resource, base_url))

    dirname = os.path.dirname(
        get_cache_path(resource, cache_dir, subdomain=subdomain, page=1)
    )
    os.makedirs(dirname, exist_ok=True)
    n_downloaded_pages = len(os.listdir(dirname))

    if force:
        n_pages_to_load = 0
    else:
        n_pages_to_load = n_downloaded_pages
        if max_pages is not None:
            n_pages_to_load = min(n_pages_to_load, max_pages)

    def cached_dl_single(page, return_npage=False, _force=False):
        fname = get_cache_path(resource, cache_dir, subdomain=subdomain, page=page)
        n_page = None
        if not _force and os.path.isfile(fname):
            df = pd.read_csv(fname)
        else:
            params["page"] = page
            result = requests.get(url, params=params).json()
            n_page = result["total_pages"]
            df = utils.clean_df(pd.json_normalize(result["objects"]))
            df.to_csv(fname, index=False)
        if return_npage:
            return df, n_page
        else:
            return df

    if verbose >= 1:
        _range = tqdm.trange
    else:
        _range = range

    stacked_df = None
    if n_pages_to_load > 0:
        if verbose >= 1:
            print(
                "Loading {} (of {}) cached pages".format(
                    n_pages_to_load, n_downloaded_pages
                )
            )

        for page in _range(1, n_pages_to_load + 1):
            df = cached_dl_single(page)
            if stacked_df is None:
                stacked_df = df
            elif page >= n_pages_to_load:
                # Don't add the final page yet, because this file may be out of
                # date. We sort by records id, which should only ever increase
                # in the database.
                pass
            else:
                stacked_df = stacked_df.append(df, ignore_index=True)

    if max_pages is not None and n_pages_to_load >= max_pages:
        # Assume our cache is up-to-date
        if page > 1:
            # Add the final page
            stacked_df = stacked_df.append(df, ignore_index=True)
        if return_updated:
            return stacked_df, False
        return stacked_df

    # Re-download the last page downloaded. This is case more content was
    # added and this is now only part of the last page.
    if verbose >= 1:
        if n_pages_to_load > 0:
            print("Re-downloading last cached page, to check for updates")
            existing_n_records_last = df["id"].iloc[-1]
        else:
            print("Downloading first page")
    page = max(1, n_pages_to_load)
    df, n_page = cached_dl_single(page, return_npage=True, _force=True)
    if stacked_df is None or n_pages_to_load == 1:
        stacked_df = df
    else:
        stacked_df = stacked_df.append(df, ignore_index=True)

    # Limit to only the pages we were asked to download
    if max_pages is not None:
        n_page = min(n_page, max_pages)

    if page >= n_page:
        # The last page loaded is the total number of pages in the database,
        # so we are done.
        if not return_updated:
            return stacked_df
        if existing_n_records_last == df["id"].iloc[-1]:
            return stacked_df, False
        else:
            return stacked_df, True

    prior_page = page

    if verbose >= 1:
        print(
            "Downloading remaining {}/{} {}pages".format(
                n_page - prior_page,
                n_page,
                "" if max_pages is None else "requested ",
            )
        )
    # Load remaining pages, saving them to the cache
    for page in _range(prior_page + 1, n_page + 1):
        df = cached_dl_single(page)
        stacked_df = stacked_df.append(df, ignore_index=True)

    if return_updated:
        return stacked_df, True
    return stacked_df


def cached_download(resource, cache_dir, pagination="auto", max_pages=None, **kwargs):
    """
    Download the entirety of a resource from SQUIDLE, caching each result page.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    cache_dir : str
        Path to cache directory.
    pagination : bool or "auto", optional
        Whether to save the cache with each page cached separately. Otherwise,
        the whole resource is cached as a single file. If `"auto"` (default),
        pagination is used unless the resource is either `"platform"` or
        `"campaign"`.
    max_pages : int or None, optional
        Maximum number of pages to download. If `None` (default), all pages are
        downloaded.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.
    return_updated : bool, optional
        Whether to return whether the cache was updated. Default is `False`.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    if pagination == "auto":
        pagination = resource not in {"platform", "campaign"}
    if pagination:
        return cached_download_paginated(
            resource, cache_dir, max_pages=max_pages, **kwargs
        )
    else:
        return cached_download_nopagination(resource, cache_dir, **kwargs)


def download_resource_to_csv_cached(
    resource, destination, cache_dir, max_pages=None, verbose=1, **kwargs
):
    """
    Download or transfer from cache the a resource from SQUIDLE into a CSV file.

    Parameters
    ----------
    resource : str
        Name of resource to download (e.g. `"platform"`, `"campaign"`,
        `"deployment"`, `"media"`, `"pose"`).
    destination : str
        Name of output file.
    cache_dir : str
        Path to cache directory.
    max_pages : int or None, optional
        Maximum number of pages to download.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    force : bool, optional
        Whether to ignore existing cached files, downloading from scratch and
        overwriting them.

    Returns
    -------
    pandas.DataFrame
        Output dataset.
    """
    t0 = time.time()

    if verbose >= 1:
        print(
            'Will download (cached) "{}" data to CSV file {}'.format(
                resource,
                destination,
            )
        )

    df, is_updated = cached_download(
        resource,
        cache_dir,
        max_pages=max_pages,
        return_updated=True,
        verbose=verbose,
        **kwargs,
    )

    if verbose >= 1:
        print("Saving output dataset to {}".format(destination))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    df.to_csv(destination, index=False)
    if verbose >= 1:
        print(
            "{} {} {} records {}in {}".format(
                "Downloaded" if is_updated else "Loaded",
                len(df),
                resource,
                "" if is_updated else "from cache ",
                datetime.timedelta(seconds=time.time() - t0),
            )
        )
        if is_updated:
            print("The cache was updated.")

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
        "--cache",
        dest="cache_dir",
        type=str,
        default=None,
        help="Path to cache directory. If unset, caching is not used.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Download even if cache exists.",
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

    if os.path.isfile(kwargs["destination"]) and not kwargs["force"]:
        print(
            "Target file {} already exists and will be overwritten.".format(
                kwargs["destination"]
            )
        )

    if kwargs["cache_dir"] is not None:
        download_resource_to_csv_cached(**kwargs)
    else:
        kwargs.pop("cache_dir")
        kwargs.pop("force")
        download_resource_to_csv(**kwargs)


if __name__ == "__main__":
    main()
