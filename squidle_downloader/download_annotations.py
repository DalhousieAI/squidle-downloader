#!/usr/bin/env python

"""
Downloading annotation sets from SQUIDLE.
"""

import datetime
import functools
import os
import sys
import time

import pandas as pd
import requests
import tqdm

from . import __meta__, utils

COLUMNS = [
    "comment",
    "id",
    "likelihood",
    "needs_review",
    "tag_names",
    "updated_at",
    "label.id",
    "label.uuid",
    "label.name",
    "label.lineage_names",
    "label.translated.id",
    "label.translated.uuid",
    "label.translated.name",
    "label.translated.lineage_names",
    "label.translated.translation_info",
    "point.media.deployment.key",
    "point.media.deployment.id",
    "point.media.deployment.name",
    "point.media.deployment.campaign.key",
    "point.media.deployment.campaign.id",
    "point.media.deployment.campaign.name",
    "point.media.key",
    "point.media.id",
    "point.media.path_best",
    "point.media.timestamp_start",
    "point.pose.data.temperature",
    "point.pose.data.salinity",
    "point.pose.data.chlorophyll_conc",
    "point.pose.data.backscatter_rat",
    "point.pose.alt",
    "point.pose.dep",
    "point.pose.lat",
    "point.pose.lon",
    "point.pose.timestamp",
    "point.data",
    "point.id",
    "point.x",
    "point.y",
    "point.t",
    "user.username",
]


def download_annotation_set(
    destination, annotation_set, api_token=None, verbose=1, print_indent=0
):
    """
    Download an annotation set as a CSV file.

    Parameters
    ----------
    destination : str
        Output path.
    annotation_set : int
        Annotation set ID.
    api_token : str or None, optional
        API token. If this is not supplied, the dataset must be publicly
        available.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is `0`.
    """
    padding = " " * print_indent

    endpoint = "https://squidle.org/api/annotation_set"
    colstr = '["' + '","'.join(COLUMNS) + '"]'
    cmd = (
        str(annotation_set)
        + "/export"
        + "?template=dataframe.csv&disposition=attachment"
        + "&include_columns="
        + colstr
        + '&f={"operations":[{"module":"pandas","method":"json_normalize"}]}'
        + '&q={"filters":[{"name":"label_id","op":"is_not_null"}]}'
        + '&translate={"vocab_registry_keys":["worms","caab","catami"],"target_label_scheme_id":"1"}'
    )
    # Translation targets
    #    2 = CATAMI 1.4
    # *  1 = SQUIDLE (extended version of CATAMI 1.4)
    # * 11 = RLS Australian Coral Species List (extends RLS Catalogue, id=8)
    #    3 = DFO Critters
    #    4 = CBiCS
    url = endpoint + "/" + cmd

    ext = os.path.splitext(destination)[1]
    if len(ext) == 0:
        destination = os.path.join(destination, "{}.csv".format(annotation_set))

    headers = {}
    if api_token is not None:
        headers["auth-token"] = api_token

    if verbose >= 1:
        print(
            "{}Downloading annotation set {} to '{}'".format(
                padding, annotation_set, destination
            )
        )

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1048576):
            f.write(chunk)
    if verbose >= 1:
        print(
            "{}Wrote annotation set {} to file '{}'".format(
                padding, annotation_set, destination
            )
        )


def list_available_annotation_sets(api_token=None, verbose=1):
    """
    Fetch ids of available annotation sets.

    Parameters
    ----------
    api_token : str or None, optional
        API token. If this is not supplied, the dataset must be publicly
        available.
    verbose : int, optional
        Verbosity level. Default is ``1``.

    Returns
    -------
    annotation_set_ids : list
        List of annotation set IDs which the user has permission to download.
    """
    # Make request
    headers = {}
    if api_token is not None:
        headers["auth-token"] = api_token

    url = "https://squidle.org/api/annotation_set"

    if verbose >= 1:
        print(
            "Fetching list of {}available annotation sets{}...".format(
                "publicly " if api_token is None else "",
                " (using your API token)" if api_token is not None else "",
            )
        )
    try:
        response = requests.get(
            url,
            headers=headers,
            params={"results_per_page": 20_000},
        )
    except requests.exceptions.RequestException as err:
        print("Error while handling: {}".format(url))
        print(err)
        return []
    if response.status_code != 200:
        if verbose >= 0:
            print(
                "Unable to fetch annotation sets {}(HTTP Status {})".format(
                    "without authentication token " if api_token is None else "",
                    response.status_code,
                )
            )
        return []
    df = pd.DataFrame.from_records(response.json()["objects"])
    ids = list(df["id"])
    if verbose >= 1:
        print(
            "Found {} {}annotation sets".format(
                len(ids),
                "public " if api_token is None else "",
            )
        )
    return ids


def download_annotation_sets(
    destination, annotation_sets=None, api_token=None, verbose=1, use_tqdm=True
):
    """
    Download multiple annotation sets.

    Parameters
    ----------
    destination : str
        Output path. If `annotation_sets` contains only a single annotation
        set, this can be an output file name. Otherwise, this must be a path
        to an output directory.
    annotation_sets : list of int or None
        Annotation set IDs. If this is not given, all annotation sets,
        available using the credentials given in `api_token`, are downloaded.
    api_token : str or None, optional
        API token. If this is not supplied, the dataset must be publicly
        available.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if writing to a file.
        Default is ``True``.
    """
    t0 = time.time()

    if annotation_sets is None:
        annotation_sets = list_available_annotation_sets(
            api_token=api_token, verbose=verbose
        )

    ext = os.path.splitext(destination)[1]
    if len(ext) > 0 and len(annotation_sets) > 1:
        raise ValueError(
            "Destination should be a directory name when downloading multiple"
            " annotation sets ({} to download), but '{}' was given.".format(
                len(annotation_sets), destination
            )
        )

    if len(annotation_sets) < 1:
        if verbose >= 0:
            print("No annotation sets to download.")
        return

    print_indent = 4
    if verbose >= 1:
        if len(annotation_sets) == 1:
            print_indent = 0
            use_tqdm = False
            verbose += 1
        else:
            print(
                "Downloading {} annotation sets to '{}'".format(
                    len(annotation_sets), destination
                )
            )

    if verbose != 1:
        use_tqdm = False

    n_download = 0
    n_error = 0
    t1 = time.time()

    for i_set, annotation_set in enumerate(
        tqdm.tqdm(annotation_sets, total=len(annotation_sets), disable=not use_tqdm)
    ):
        if i_set > n_error and (
            verbose >= 3
            or (verbose >= 1 and not use_tqdm and (i_set <= 5 or i_set % 100 == 0))
        ):
            t_elapsed = time.time() - t1
            if n_download > 0:
                t_remain = t_elapsed / n_download * (len(annotation_sets) - i_set)
            else:
                t_remain = t_elapsed / i_set * (len(annotation_sets) - i_set)
            print(
                "Processed {:4d}/{} annotation sets ({:6.2f}%) in {} (approx. {} remaining)"
                "".format(
                    i_set,
                    len(annotation_sets),
                    100 * i_set / len(annotation_sets),
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                ),
                flush=True,
            )

        try:
            download_annotation_set(
                destination,
                annotation_set,
                api_token=api_token,
                verbose=verbose - 1,
                print_indent=print_indent,
            )
            n_download += 1

        except (
            requests.exceptions.RequestException,
            requests.exceptions.HTTPError,
        ) as err:
            print("Error while handling annotation set {}:".format(annotation_set))
            print(err)
            n_error += 1
            continue

    if verbose >= 1:
        message = "Finished processing {} annotation sets in {}.".format(
            len(annotation_sets), datetime.timedelta(seconds=time.time() - t0)
        )
        if n_error > 0:
            message += " There {} {} download error{}.".format(
                "was" if n_error == 1 else "were",
                n_error,
                "" if n_error == 1 else "s",
            )
        print(message, flush=True)


def get_parser():
    """
    Build CLI parser for downloading SQUIDLE annotation sets.

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
        description="Download an annotation set from SQUIDLE to CSV file(s).",
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
        help="""
            Output path. This must be a directory when downloading multiple
            annotation sets, but can be a single CSV file when only downloading
            one annotation set.
        """,
    )
    parser.add_argument(
        "--annotation-set",
        dest="annotation_sets",
        nargs="*",
        type=int,
        help="""
            Annotation set ID. If this is not given, all annotation sets
            available are downloaded.
        """,
    )
    parser.add_argument(
        "--api-token",
        type=str,
        help="API token.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
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
    Run command line interface for downloading annotation sets.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())
    kwargs["verbose"] -= kwargs.pop("quiet", 0)
    download_annotation_sets(**kwargs)


if __name__ == "__main__":
    main()
