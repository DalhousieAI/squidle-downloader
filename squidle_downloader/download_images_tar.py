#!/usr/bin/env python

"""
Downloading images listed in a SQUIDLE formatted CSV file to tarfiles.
"""

import datetime
import functools
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import PIL.Image
import requests
import tqdm

from . import __meta__, utils


def download_images(
    df,
    tar_fname,
    convert_to_jpeg=False,
    jpeg_quality=95,
    skip_existing=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
):
    """
    Download images into a tarball.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of images to download.
    tar_fname : str
        Path to output tarball file.
    convert_to_jpeg : bool, optional
        Whether to convert PNG images to JPEG format. Default is `False`.
    jpeg_quality : int or float, optional
        Quality to use when converting to JPEG. Default is `95`.
    skip_existing : bool, optional
        Whether to skip existing outputs. If `False`, an error is raised when
        the destination already exists. Default is `True`.
    verbose : int, optional
        Verbosity level. Default is `1`.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if writing to a file.
        Default is `True`.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is `0`.

    Returns
    -------
    None
    """
    padding = " " * print_indent
    innerpad = padding + " " * 4
    if verbose >= 1:
        print(
            padding
            + "Downloading {} images into tarball {}".format(len(df), tar_fname),
            flush=True,
        )

    if verbose == 1 and use_tqdm:
        maybe_tqdm = functools.partial(tqdm.tqdm, total=len(df))
    else:
        use_tqdm = False
        maybe_tqdm = lambda x: x  # noqa: E731

    n_already_downloaded = 0
    n_download = 0
    n_error = 0

    t0 = time.time()

    os.makedirs(os.path.dirname(tar_fname), exist_ok=True)

    try:
        with tarfile.open(tar_fname, mode="a") as tar:
            pass
    except tarfile.ReadError:
        t_wait = 5
        if verbose >= 1:
            print(
                "{}Unable to open {}.\n{}File {} will be deleted in {} seconds..."
                "".format(padding, tar_fname, padding, tar_fname, t_wait),
            )
            print("{}Deleting in".format(padding), end="", flush=True)
            for i in range(t_wait // 1):
                print(" {}...".format(t_wait - i), end="", flush=True)
                time.sleep(1)
            print(" Deleting!")
            if t_wait % 1 > 0:
                time.sleep(t_wait % 1)
        else:
            time.sleep(5)
        os.remove(tar_fname)
        if verbose >= 1:
            print("{}Existing file {} deleted".format(padding, tar_fname), flush=True)

    for i_row, (_, row) in enumerate(maybe_tqdm(df.iterrows())):
        if pd.isna(row["url"]) or row["url"] == "":
            n_error += 1
            if verbose >= 2:
                print(
                    "{}Missing URL for entry\n{}".format(innerpad, row),
                    flush=True,
                )
            continue

        if i_row > n_error and (
            verbose >= 3
            or (verbose >= 1 and not use_tqdm and (i_row <= 5 or i_row % 100 == 0))
        ):
            t_elapsed = time.time() - t0
            t_remain = t_elapsed / i_row * (len(df) - i_row)
            print(
                "{}Processed {:4d}/{} urls ({:6.2f}%) in {} (approx. {} remaining)"
                "".format(
                    padding,
                    i_row,
                    len(df),
                    100 * i_row / len(df),
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                ),
                flush=True,
            )

        destination = row["key"].strip()
        ext = os.path.splitext(destination)[1]
        expected_ext = os.path.splitext(row["url"].rstrip(" /"))[1]
        if expected_ext and ext.lower() != expected_ext.lower():
            destination += expected_ext
        destination = os.path.join(row["deployment"], destination)
        ext = os.path.splitext(destination)[1]
        if convert_to_jpeg and ext.lower() not in {".jpg", ".jpeg"}:
            needs_conversion = True
            destination = os.path.splitext(destination)[0] + ".jpg"
        else:
            needs_conversion = False

        with tarfile.open(tar_fname, mode="a") as tar:
            if destination in tar.getnames():
                if not skip_existing:
                    raise EnvironmentError(
                        "Destination {} already exists within {}".format(
                            destination, tar_fname
                        )
                    )
                if verbose >= 3:
                    print(innerpad + "Already downloaded {}".format(destination))
                n_already_downloaded += 1
                continue

            try:
                r = requests.get(row["url"].strip(), stream=True)
            except requests.exceptions.RequestException as err:
                print("Error handing: {}".format(row["url"]))
                print(err)
                n_error += 1
                continue
            if r.status_code != 200:
                if verbose >= 1:
                    print(
                        innerpad
                        + "Bad URL (HTTP Status {}): {}".format(
                            r.status_code, row["url"]
                        )
                    )
                n_error += 1
                continue

            if verbose >= 3:
                print(innerpad + "Downloading {}".format(row["url"]))
            with tempfile.TemporaryDirectory() as dir_tmp:
                fname_tmp = os.path.join(
                    dir_tmp,
                    os.path.basename(row["url"].rstrip(" /")),
                )

                with open(fname_tmp, "wb") as f:
                    for chunk in r:
                        f.write(chunk)
                if verbose >= 4:
                    print(innerpad + "  Wrote to {}".format(fname_tmp))

                if needs_conversion:
                    fname_tmp_new = os.path.join(
                        dir_tmp,
                        os.path.basename(destination),
                    )
                    if verbose >= 4:
                        print(
                            innerpad
                            + "  Converting {} to JPEG {}".format(
                                fname_tmp, fname_tmp_new
                            )
                        )
                    im = PIL.Image.open(fname_tmp)
                    im = im.convert("RGB")
                    im.save(fname_tmp_new, quality=jpeg_quality)
                    fname_tmp = fname_tmp_new

                if verbose >= 4:
                    print(
                        innerpad
                        + "  Adding {} to archive as {}".format(fname_tmp, destination)
                    )
                tar.add(fname_tmp, arcname=destination)
        n_download += 1

    if verbose >= 1:
        print(padding + "Finished processing {} images".format(len(df)))
        s = "{}{} {} image{} already downloaded.".format(
            padding,
            "All" if n_already_downloaded == len(df) else "There were",
            n_already_downloaded,
            "" if n_already_downloaded == 1 else "s",
        )
        if n_error > 0:
            s += " There {} {} download error{}.".format(
                "was" if n_error == 1 else "were",
                n_error,
                "" if n_error == 1 else "s",
            )
        if n_download > 0:
            s += " The remaining {} image{} downloaded.".format(
                n_download,
                " was" if n_download == 1 else "s were",
            )
        print(s, flush=True)
    return


def download_images_by_campaign(
    df,
    output_dir,
    skip_existing=True,
    i_proc=None,
    n_proc=None,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
    **kwargs
):
    """
    Download all images from a DataFrame into tarfiles, one for each campaign.

    Parameters
    ----------
    input_csv : str
        Path to CSV file.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to silently skip downloading files for which the destination
        already exist. Otherwise an error is raised. Default is `True`.
    i_proc : int or None, optional
        Run on only a subset of the campaigns in the CSV file. If `None`
        (default), the entire dataset will be downloaded by this process.
        Otherwise, `n_proc` must also be set, and `1/n_proc`-th of the
        campaigns will be processed.
    n_proc : int or None, optional
        Number of processes being run. Default is `None`.
    verbose : int, optional
        Verbosity level. Default is `1`.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if writing to a file.
        Default is `True`.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is `0`.
    **kwargs
        Additional keword arguments as per ``download_images``.

    Returns
    -------
    None
    """
    padding = " " * print_indent

    t0 = time.time()

    if (i_proc is not None and n_proc is None) or (
        i_proc is None and n_proc is not None
    ):
        raise ValueError(
            "Both `i_proc` and `n_proc` must be defined when partitioning"
            " the CSV file."
        )

    if verbose >= 2 and not skip_existing:
        print("Warning: Existing outputs will result in an error.", flush=True)

    campaign2idx = utils.unique_map(df["campaign"])

    campaigns_to_process = sorted(campaign2idx)

    if n_proc:
        partition_size = len(campaign2idx) / n_proc
        i_proc == 0 if i_proc == n_proc else i_proc
        start_idx = round(i_proc * partition_size)
        end_idx = round((i_proc + 1) * partition_size)
        campaigns_to_process = campaigns_to_process[start_idx:end_idx]

    n_to_process = len(campaigns_to_process)

    if verbose >= 1:
        print(
            "{}There are {} campaigns in the CSV file.".format(
                padding, len(campaign2idx)
            ),
            flush=True,
        )
        if n_proc:
            print(
                "{}Worker {} of {}. Will process {} campaigns.".format(
                    padding, i_proc, n_proc, n_to_process
                )
            )

    if verbose == 1 and use_tqdm:
        maybe_tqdm = tqdm.tqdm
    else:
        use_tqdm = False
        maybe_tqdm = lambda x: x  # noqa: E731

    for i_campaign, campaign in enumerate(maybe_tqdm(campaigns_to_process)):
        if not n_proc:
            pass

        if verbose >= 1 and not use_tqdm and i_campaign > 0:
            t_elapsed = time.time() - t0
            t_remain = t_elapsed / i_campaign * (n_to_process - i_campaign)
            print(
                "{}Processed {:3d}/{} campaigns ({:6.2f}%) in {} (approx. {} remaining)"
                "".format(
                    padding,
                    i_campaign,
                    n_to_process,
                    100 * i_campaign / n_to_process,
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                )
            )
        if verbose >= 1 and not use_tqdm:
            print(
                '{}Processing campaign "{}" ({}/{})'.format(
                    padding, campaign, i_campaign, len(campaign2idx)
                ),
                flush=True,
            )

        subdf = df.loc[campaign2idx[campaign]]
        tar_fname = os.path.join(output_dir, campaign + ".tar")

        download_images(
            subdf,
            tar_fname,
            skip_existing=skip_existing,
            verbose=verbose - 1,
            use_tqdm=use_tqdm,
            print_indent=print_indent + 4,
            **kwargs,
        )

    if verbose >= 1:
        print(
            "Processed {} campaigns in {}".format(
                n_to_process,
                datetime.timedelta(seconds=time.time() - t0),
            ),
            flush=True,
        )
    return


def download_images_by_campaign_from_csv(
    input_csv,
    output_dir,
    *args,
    skip_existing=True,
    verbose=1,
    **kwargs,
):
    """
    Download all images from a CSV file into tarfiles, one for each campaign.

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
    **kwargs
        Additional arguments as per ``download_images_by_campaign``.

    Returns
    -------
    None
    """
    t0 = time.time()

    if verbose >= 1:
        print(
            "Will download images listed in {} into tarfiles by campaign name"
            "\nInto directory: {}".format(
                input_csv,
                output_dir,
            )
        )
        if skip_existing:
            print("Existing outputs will be skipped.")
        else:
            print("Existing outputs will generate an error.")
        print("Reading CSV file ({})...".format(utils.file_size(input_csv)), flush=True)
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
    )
    if verbose >= 1:
        print("Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True)

    ret = download_images_by_campaign(
        df, output_dir, *args, skip_existing=skip_existing, verbose=verbose, **kwargs
    )

    if verbose >= 1:
        print(
            "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
            flush=True,
        )
    return ret


def get_parser():
    """
    Build CLI parser for downloading a dataset into tarfiles by campaign.

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
        description="""
            Download all images listed in a SQUIDLE CSV file into tarfiles.

            A tarfile (a.k.a. tarball) is created for each campaign. Within
            the tarfile, a directory for each deployment is created.
            Each image that is part of that campaign is downloaded as
            deployment/key.jpg within the corresponding tarfile named as
            output_dir/campaign.tar.
        """,
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
        "--jpeg",
        dest="convert_to_jpeg",
        action="store_true",
        help="Convert images to JPEG format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=float,
        default=95,
        help="Quality to use when converting to JPEG. Default is %(default)s.",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--nproc",
        dest="n_proc",
        metavar="NPROC",
        type=int,
        help="Number of processing partitions being run.",
    )
    parser.add_argument(
        "--iproc",
        "--proc",
        dest="i_proc",
        metavar="IPROC",
        type=int,
        help="Partition index for this process.",
    )
    parser.add_argument(
        "--fail-existing",
        dest="skip_existing",
        action="store_false",
        help="""
            Raise an error if any outputs already exist in the target tarfile.
            Default behaviour is to quietly skip processing any existing files.
        """,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=2,
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
    Run command line interface for downloading images to tarfiles by campaign.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_images_by_campaign_from_csv(**kwargs)


if __name__ == "__main__":
    main()
