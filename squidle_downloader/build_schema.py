#!/usr/bin/env python

"""
Building consolidated label schema for SQUIDLE data.
"""

import copy
import datetime
import functools
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import tqdm

from . import __meta__, utils

URL_CATAMI = "https://squidle.org/api/label_scheme_file/2/data"


def get_catami_scheme():
    """
    Download CATAMI v1.4 labelling scheme.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        The CATAMI v1.4 scheme.

    Note
    ----
    The original scheme has an error in the parent IDs for the "Seagrasses"
    children: "Elliptical leaves" and "Strap-like leaves". This error is fixed
    in the result, so the parent IDs are the "Seagrasses" ID instead.
    """
    # Load the CATAMI v1.4 scheme, as hosted on SQUIDLE.
    # Note that the official file is located at
    # https://data.pawsey.org.au/public/?path=/WA%20Node%20Ocean%20Data%20Network/AODN/catami/catami-caab-codes_1.4.csv
    # however, the official source requires us to open the page with a browser
    # and click a button in order to get a download link.
    # The one on SQUIDLE can be downloaded in an automated manner.

    df = pd.read_csv(
        URL_CATAMI,
        skiprows=1,
        dtype={
            "SPECIES_CODE": str,
            "CATAMI_DISPLAY_NAME": str,
            "CATAMI_PARENT_ID": str,
            "CPC_CODES": str,
            "Changes since v1.2": str,
            "Changes since v1.3": str,
        },
    )

    # Fix bug in official CSV file.
    # Seagrasses: Elliptical leaves
    # Seagrasses: Strap-like leaves
    df.at[df["SPECIES_CODE"] == "63600902", "CATAMI_PARENT_ID"] = "63600901"
    df.at[df["SPECIES_CODE"] == "63600903", "CATAMI_PARENT_ID"] = "63600901"

    # Add a SOURCE column, showing which version of CATAMI the field was
    # introduced or last updated
    df["SOURCE"] = "v1.2"
    df.at[
        (~df["Changes since v1.2"].isna()) & (df["Changes since v1.2"] != ""), "SOURCE"
    ] = "v1.3"
    df.at[
        (~df["Changes since v1.3"].isna()) & (df["Changes since v1.3"] != ""), "SOURCE"
    ] = "v1.4"
    df.drop(columns=["Changes since v1.2", "Changes since v1.3"], inplace=True)

    return df


def fix_repeated_codes(df, inplace=True, verbose=1):
    """
    Fix repeated CATAMI codes, with hierarchy inferred based on display name.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with CATAMI headers.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    pandas.DataFrame
    """
    if not inplace:
        df = df.copy()

    n_reps = df.value_counts(subset="SPECIES_CODE")

    for species_code, n_rep in n_reps[n_reps > 1].items():
        if verbose > 0:
            print(
                "Fixing {} repetitions of SPECIES_CODE {}".format(n_rep, species_code)
            )
        tally = defaultdict(lambda: 0)
        tally[1] = 1
        new_codes = {}
        df_repeats = df[df["SPECIES_CODE"] == species_code]
        primary_row = df_repeats.iloc[0]
        primary_name = primary_row["CATAMI_DISPLAY_NAME"]
        primary_depth = primary_name.count(":")

        any_updated = True
        some_missing = True
        while any_updated and some_missing:
            any_updated = False
            some_missing = False
            for i_row, row in df_repeats[1:].iterrows():
                depth = row["CATAMI_DISPLAY_NAME"].count(":") - primary_depth
                if depth < 1:
                    print("Warning: Multiple rows at the same shallowest depth:")
                    print()
                    print(primary_row)
                    print()
                    print(row)
                    continue
                if depth == 1:
                    tally[1] += 1
                    df.at[i_row, "SPECIES_CODE"] += "_{}".format(tally[1])
                    new_codes[row["CATAMI_DISPLAY_NAME"]] = df.at[i_row, "SPECIES_CODE"]
                    any_updated = True
                    continue
                parent = row["CATAMI_DISPLAY_NAME"][
                    : row["CATAMI_DISPLAY_NAME"].rfind(":")
                ]
                if parent not in new_codes:
                    some_missing = True
                new_codes[row["CATAMI_DISPLAY_NAME"]] = new_codes[
                    parent
                ] + "_{}".format(tally[parent])
                df.at[i_row, "SPECIES_CODE"] = new_codes[row["CATAMI_DISPLAY_NAME"]]
                df.at[i_row, "CATAMI_PARENT_ID"] = new_codes[parent]
                tally[parent] += 1
                any_updated = True
    return df


def get_squidle_extended_scheme(
    catami_format=True, fixup=True, include_v1=True, include_plus=True
):
    """
    Download the SQUIDLE+ extended label scheme.

    This label scheme is extended from the Squidle 1.0 label scheme, which was
    itself an extension of CATAMI v1.4.

    Parameters
    ----------
    catami_format : bool, optional
        Whether to reformat the scheme so its columns have the same names as
        those used in the CATAMI v1.4 CSV. Otherwise, the format will be as
        per the extended SQUIDLE+ scheme. Default is `True`.
    fixup : bool, optional
        Whether to fix typos. Default is `True`.
    include_v1 : bool, optional
        Whether to download the SQUIDLE 1.0 scheme from both
        https://squidle.org and https://soi.squidle.org.
    include_plus : bool, optional
        Whether to download the SQUIDLE+ scheme from https://soi.squidle.org

    Returns
    -------
    pandas.DataFrame
        The SQUIDLE+ extended label scheme.
    """

    def _add_catami_parent_field(df):
        # Add the missing CATAMI_PARENT_ID field, based on the "id" and "parent_id"
        # fields
        for i_row, row in df.iterrows():
            if np.isnan(row["parent_id"]):
                continue
            parents = df[df["id"] == row["parent_id"]]
            if len(parents) == 0:
                print(
                    "Can't find parent with ID {} for {}".format(
                        row["parent_id"], row["code_name"]
                    )
                )
                continue
            if len(parents) > 1:
                print(
                    "Multiple parents with ID {} found for {}".format(
                        row["parent_id"], row["code_name"]
                    )
                )
            df.at[i_row, "CATAMI_PARENT_ID"] = parents.iloc[0]["caab_code"]
        return df

    # We use the latest version of the scheme files from SOI SQUIDLE
    # We're using the original tag scheme files for this instead of downloading
    # the current live tag scheme because the files contain extra fields
    # caab_code and cpc_code that we'll make use of to harmonise the scheme
    # with CATAMI v1.4.
    dtypes = {
        "caab_code": str,
        "code_name": str,
        "cpc_code": str,
        "id": float,  # Can't cast NaN to int
        "parent_id": float,  # Can't cast NaN to int
        "point_colour": str,
        "name": str,
        "full_name": str,
        "short_code": str,
        "date_added": str,
    }
    dfs = []
    if include_v1:
        # This first scheme is SQUIDLE 1.0 from https://squidle.org/
        with warnings.catch_warnings():
            # Ignore incorrect file extension warning
            warnings.simplefilter("ignore")
            df = pd.read_excel(
                "https://squidle.org/api/label_scheme_file/1/data",
                engine="openpyxl",
                dtype=dtypes,
            )
        df["SOURCE"] = "SQUIDLE 1.0"
        if catami_format:
            df = _add_catami_parent_field(df)
        dfs.append(df)

        # Download SQUIDLE 1.0 from https://soi.squidle.org/
        result = requests.get(
            "https://soi.squidle.org/api/tag_scheme_file/get/4"
        ).json()
        df = pd.DataFrame.from_records(result["objects"])
        df["SOURCE"] = "SQUIDLE 1.0 (SOI)"
        df.drop(columns=["description", "parent", "resource_uri"], inplace=True)
        if catami_format:
            df = _add_catami_parent_field(df)
        dfs.append(df)
    if include_plus:
        # This scheme is an extension of the SQUIDLE 1.0 scheme (as seen on the
        # base SQUIDLE URL). It split over two files because it was itself
        # extended; the second file (.csv) is an extension to the first (.xlsx).
        with warnings.catch_warnings():
            # Ignore incorrect file extension warning
            warnings.simplefilter("ignore")
            df_a = pd.read_excel(
                "https://soi.squidle.org/api/tag_scheme_file/get/5",
                engine="openpyxl",
                dtype=dtypes,
            )
        df_a["SOURCE"] = "SQUIDLE+"

        # Download extension file
        df_b = pd.read_csv(
            "https://soi.squidle.org/api/tag_scheme_file/get/15",
            dtype=dtypes,
        )
        df_b["SOURCE"] = "SQUIDLE+"
        # Make the fields in the "part 2" extension line up with the other parts
        df_b.drop(columns=["date_added"], inplace=True)
        df_b.rename(
            columns={"full_name": "code_name", "short_code": "cpc_code"}, inplace=True
        )

        if catami_format:
            df = pd.concat([df_a, df_b], ignore_index=True)
            df = _add_catami_parent_field(df)
            dfs.append(df)
        else:
            dfs.append(df_a)
            dfs.append(df_b)

    # Merge the scheme files together
    df_sqdx = pd.concat(dfs, ignore_index=True)
    df_sqdx.drop(columns=["point_colour"], inplace=True)
    if fixup:
        for col in ["code_name"]:
            fixed_col = df_sqdx[col]
            fixed_col = fixed_col.str.replace("Psammacora", "Psammocora")
            fixed_col = fixed_col.str.replace("  ", " ")
            fixed_col = fixed_col.str.replace(" spp.", " spp")
            fixed_col = fixed_col.str.replace("Not labeled yet...", "Not labeled yet")
            fixed_col = fixed_col.str.replace("Unscorable.", "Unscorable")
            df_sqdx[col] = fixed_col
        for col in ["caab_code", "code_name", "cpc_code"]:
            df_sqdx[col] = df_sqdx[col].str.strip()

    df_sqdx.drop_duplicates(subset=["cpc_code", "code_name"], inplace=True)
    if not catami_format:
        # If we're using the SQUIDLE field name format, we're done at this
        # point. Otherwise, there is a bit more work to harmonise with CATAMI.
        return df_sqdx
    # Convert the squidle extended xlsx to have fields in the CATAMI format
    df_sqdx.rename(
        columns={
            "caab_code": "SPECIES_CODE",
            "code_name": "CATAMI_DISPLAY_NAME",
            "cpc_code": "CPC_CODES",
        },
        inplace=True,
    )
    # Fix repeated 72000901	codes for new Bacterial mats subclasses
    df_sqdx = fix_repeated_codes(df_sqdx)
    # Remove superfluous fields
    df_sqdx.drop(columns=["id", "parent_id", "name"], inplace=True)
    return df_sqdx


def get_amc_scheme(catami_format=True, fixup=True):
    """
    Download the Australian Morphospecies Catalogue label scheme.

    This label scheme is extended from CATAMI v1.4.

    Parameters
    ----------
    catami_format : bool, optional
        Whether to reformat the scheme so its columns have the same names as
        those used in the CATAMI v1.4 CSV. Otherwise, the format will be as
        per the Australian Morphospecies Catalogue scheme. Default is `True`.
    fixup : bool, optional
        Whether to fix cells which are apparent typos. Default is `True`.

    Returns
    -------
    pandas.DataFrame
        The Australian Morphospecies Catalogue label scheme.
    """
    # We're using the original tag scheme files for this instead of downloading
    # the current live tag scheme because the files contain extra fields
    # caab_code and cpc_code that we'll make use of to harmonise the scheme
    # with CATAMI v1.4.
    # The scheme is split over two files because it was itself extended;
    # the second csv file is an extension to the first, though with fewer
    # columns.
    dtypes = {
        "ID": str,
        "CAAB_CODE": str,
        "DISPLAY_NAME": str,
        "PARENT_ID": str,
        "SHORT_CODE": str,
    }
    df_amc = pd.concat(
        [
            pd.read_csv(
                "https://squidle.org/api/label_scheme_file/7/data", dtype=dtypes
            ),
            pd.read_csv(
                "https://squidle.org/api/label_scheme_file/19/data", dtype=dtypes
            ),
        ],
        ignore_index=True,
    )
    # Remove unnecessary columns
    df_amc = df_amc[["ID", "CAAB_CODE", "DISPLAY_NAME", "PARENT_ID", "SHORT_CODE"]]
    # There should be no duplicates to drop
    df_amc.drop_duplicates(subset="DISPLAY_NAME", inplace=True)

    if fixup:
        # Appears that entry
        # 11168915_6; Cnidaria: Corals: Black & Octocorals: Fan (2D):
        #               Fern-Frond: Complex: Blackish Red Complex Fern
        # has its sibling
        # 11168915_2; Cnidaria: Corals: Black & Octocorals: Fan (2D):
        #               Fern-Frond: Complex: Gorgonian Pink
        # marked as its parent, instead of the sibling's parent
        # (which appears to be its true parent)
        # 11168915_1; Cnidaria: Corals: Black & Octocorals: Fan (2D):
        #               Fern-frond: Complex
        df_amc.at[df_amc["PARENT_ID"] == "11168915_2", "PARENT_ID"] = "11168915_1"
        for col in ["CAAB_CODE", "DISPLAY_NAME"]:
            df_amc[col] = df_amc[col].str.strip()

    if not catami_format:
        # If we're using the AMC field name format, we're done at this
        # point. Otherwise, there is a bit more work to harmonise with CATAMI.
        return df_amc

    # Convert the AMC fields to be named as per the CATAMI format
    df_amc.rename(
        columns={
            "ID": "SPECIES_CODE",
            "DISPLAY_NAME": "CATAMI_DISPLAY_NAME",
            "SHORT_CODE": "CPC_CODES",
            "PARENT_ID": "CATAMI_PARENT_ID",
        },
        inplace=True,
    )
    # The current CAAB_CODE field is not adequate because it contains
    # duplicates - entries which are expanded from CATAMI have the same codes
    # as their parent.
    df_amc.drop(columns=["CAAB_CODE"], inplace=True)
    # The current ID fields are almost the CATAMI IDs, but not quite. If they
    # for entries identical in CATAMI, they are padded with "_1". If they
    # new to AMC, they are padded with "_2", "_3", etc.
    df_amc["SPECIES_CODE"] = df_amc["SPECIES_CODE"].str.replace("_1", "")
    df_amc["CATAMI_PARENT_ID"] = df_amc["CATAMI_PARENT_ID"].str.replace("_1", "")
    return df_amc


def catami_add_lineage(
    df, inplace=True, return_construction=False, preconstructed_lineages=None
):
    """
    Add a lineage column to a DataFrame in CATAMI format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a label scheme with columns named in the same way
        as the CATAMI label scheme.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.
    return_construction : bool, optional
        Whether to return the construction dictionary. Default is `False`.
    preconstructed_lineages : dict or None, optional
        Pre-constructed set of lineage mappings to initialise.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with `"LINEAGE"` and `"NAME"` columns added.
    """
    if not inplace:
        df = df.copy()

    df["LINEAGE"] = ""
    df["NAME"] = ""

    if preconstructed_lineages is None:
        constructed_lineages = {}
    else:
        constructed_lineages = copy.deepcopy(preconstructed_lineages)

    for i_row, row in df[df["CATAMI_PARENT_ID"].isna()].iterrows():
        constructed_lineages[row["SPECIES_CODE"]] = row["CATAMI_DISPLAY_NAME"]
        df.loc[i_row, "LINEAGE"] = constructed_lineages[df.loc[i_row, "SPECIES_CODE"]]
        df.loc[i_row, "NAME"] = constructed_lineages[df.loc[i_row, "SPECIES_CODE"]]

    any_updated = True
    any_missing = True
    while any_updated and any_missing:
        any_updated = False
        any_missing = False
        for i_row, row in df.iterrows():
            if row["LINEAGE"]:
                if row["SPECIES_CODE"] not in constructed_lineages:
                    constructed_lineages[row["SPECIES_CODE"]] = constructed_lineages[
                        row["LINEAGE"]
                    ]
                    any_updated = True
                continue
            if row["SPECIES_CODE"] in constructed_lineages:
                df.loc[i_row, "LINEAGE"] = constructed_lineages[
                    df.loc[i_row, "SPECIES_CODE"]
                ]
                any_updated = True
                continue
            parent_rows = df[df["SPECIES_CODE"] == row["CATAMI_PARENT_ID"]]
            if len(parent_rows) < 1:
                print(
                    "Warning: missing parent {} for row:".format(
                        row["CATAMI_PARENT_ID"]
                    )
                )
                print(row)
                continue
            if len(parent_rows) > 1:
                print(
                    'Warning: multiple parents with code "{}" for row:'.format(
                        row["CATAMI_PARENT_ID"]
                    )
                )
                print(row)
                pass
            parent_row = df[df["SPECIES_CODE"] == row["CATAMI_PARENT_ID"]].iloc[0]
            if parent_row["SPECIES_CODE"] not in constructed_lineages:
                any_missing = True
                continue
            constructed_lineages[row["SPECIES_CODE"]] = "{} > {}".format(
                constructed_lineages[parent_row["SPECIES_CODE"]],
                row["CATAMI_DISPLAY_NAME"].split(":")[-1].strip(),
            )
            df.loc[i_row, "LINEAGE"] = constructed_lineages[
                df.loc[i_row, "SPECIES_CODE"]
            ]
            any_updated = True

    if return_construction:
        return df, constructed_lineages
    return df


def catami_add_lastnames(df, inplace=True):
    """
    Add a NAME column containing the last component of CATAMI_DISPLAY_NAME.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a label scheme with columns named in the same way
        as the CATAMI label scheme.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with `"NAME"` columns added.
    """
    if not inplace:
        df = df.copy()

    for i_row, row in df.iterrows():
        df.loc[i_row, "NAME"] = row["CATAMI_DISPLAY_NAME"].split(":")[-1].strip()

    return df


def build_squidle_lineage(rows, strip=".", parent_lineage=""):
    """
    Build the lineage for SQUIDLE label records.

    Parameters
    ----------
    rows : list of dict
        JSON formatted records, with fields `"children"`
    parent_lineage : str, optional
        Base lineage, used for recursive lineage construction. Default is `""`.

    Returns
    -------
    list of dict
        Each record has fields `"LINEAGE"` and `"NAME"` only.
    """
    output_rows = []
    for row in rows:
        if strip:
            row["name"] = row["name"].strip(strip)
        this_lineage = row["name"]
        if parent_lineage:
            this_lineage = "{} > {}".format(parent_lineage, this_lineage)
        output_rows.append(
            {
                "NAME": row["name"],
                "LINEAGE": this_lineage,
            }
        )
        if row.get("children"):
            output_rows += build_squidle_lineage(
                row["children"], strip=strip, parent_lineage=this_lineage
            )
    return output_rows


def lineage2catami_name(df, clobber=False, inplace=True):
    """
    Infer CATAMI_DISPLAY_NAME from LINEAGE.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    clobber : bool, optional
        Whether to overwrite existing CATAMI_DISPLAY_NAME entries. If `False`
        (default) only NaN and empty string values will be overwritten.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.

    Returns
    -------
    pandas.DataFrame
        The input `df`, but with entries for the `"CATAMI_DISPLAY_NAME"` column
        updated.
    """
    if not inplace:
        df = df.copy()
    if clobber:
        df_loop = df
    else:
        df_loop = df[
            df["CATAMI_DISPLAY_NAME"].isna() | (df["CATAMI_DISPLAY_NAME"] == "")
        ]
    for i_row, row in df_loop.iterrows():
        name = row["LINEAGE"].replace(" > ", ": ")
        name = utils.remove_prefix(name, "Biota: ")
        name = utils.remove_prefix(name, "Physical: ")
        df.at[i_row, "CATAMI_DISPLAY_NAME"] = name
    return df


def catami_name2lineage(df, clobber=False, inplace=True):
    """
    Infer LINEAGE from CATAMI_DISPLAY_NAME.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    clobber : bool, optional
        Whether to overwrite existing LINEAGE entries. If `False`
        (default) only NaN and empty string values will be overwritten.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.

    Returns
    -------
    pandas.DataFrame
        The input `df`, but with entries for the `"LINEAGE"` column
        updated.
    """
    if not inplace:
        df = df.copy()
    if clobber:
        df_loop = df
    else:
        df_loop = df[df["LINEAGE"].isna() | (df["LINEAGE"] == "")]
    for i_row, row in df_loop.iterrows():
        name = row["CATAMI_DISPLAY_NAME"].replace(": ", " > ")
        if not name.startswith("Biota") and not name.startswith("Physical"):
            # Going to guess it is a Biota entry
            name = "Biota > " + name
        df.at[i_row, "LINEAGE"] = name
    return df


def add_missing_codes(df, add_missing_codes=True, inplace=True, verbose=1):
    """
    Add missing CATAMI_PARENT_ID codes based on LINEAGE.

    Missing SPECIES_CODE values will also be generated.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with CATAMI headers.
    add_missing_codes : bool, optional
        Whether to also create species codes where they are missing. Default is `True`.
    inplace : bool, optional
        Whether to change the input DataFrame. Default is `True`.
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    pandas.DataFrame
    """
    if not inplace:
        df = df.copy()

    df_without_code = df[df["SPECIES_CODE"].isna() | (df["SPECIES_CODE"] == "")]
    df_with_code = df[~df["SPECIES_CODE"].isna() & (df["SPECIES_CODE"] != "")]

    # Copy over SPECIES_CODE from duplicates if there are any dupes to borrow from
    for i_row, row in df_without_code.iterrows():
        matches = df_with_code[df_with_code["LINEAGE"] == row["LINEAGE"]]
        if len(matches) > 0:
            df.at[i_row, "SPECIES_CODE"] = matches.iloc[0]["SPECIES_CODE"]
            if verbose >= 2:
                print(
                    "  Copying SPECIES_CODE {} from duplicate with LINEAGE = {}"
                    "".format(
                        matches.iloc[0]["SPECIES_CODE"], matches.iloc[0]["LINEAGE"]
                    )
                )

    df_missing_code = df[
        df["CATAMI_PARENT_ID"].isna()
        | (df["CATAMI_PARENT_ID"] == "")
        | df["SPECIES_CODE"].isna()
        | (df["SPECIES_CODE"] == "")
    ]

    if verbose >= 1:
        print(
            "Will try to add CATAMI_PARENT_ID for {} records".format(
                len(df_missing_code)
            )
        )

    any_updated = True
    any_missing = True
    while any_updated and any_missing:
        any_updated = False
        any_missing = False
        for i_row, row in df_missing_code.iterrows():
            needs_parent = pd.isnull(df.at[i_row, "CATAMI_PARENT_ID"]) or (
                df.at[i_row, "CATAMI_PARENT_ID"] == ""
            )
            needs_code = pd.isnull(df.at[i_row, "SPECIES_CODE"]) or (
                df.at[i_row, "SPECIES_CODE"] == ""
            )
            if (not needs_parent) and (not needs_code):
                # Already done this one in a previous loop
                continue
            parent_lineage = row["LINEAGE"][: row["LINEAGE"].rfind(" > ")]
            parents = df[df["LINEAGE"] == parent_lineage]
            if len(parents) <= 0:
                # This one doesn't have any parents and can't be completed
                continue
            if len(parents) > 1:
                print(
                    "Warning: More than one row with lineage {}".format(parent_lineage)
                )
            parent_row = parents.iloc[0]
            if (
                pd.isnull(parent_row["SPECIES_CODE"])
                or parent_row["SPECIES_CODE"] == ""
            ):
                # The parent hasn't had its species code added yet
                any_missing = True
                continue
            if needs_parent:
                # Set the parent ID
                df.at[i_row, "CATAMI_PARENT_ID"] = parent_row["SPECIES_CODE"]
                any_updated = True
            if (not needs_code) or (not add_missing_codes):
                # Don't need to set SPECIES_CODE for this one
                continue
            siblings = df[df["CATAMI_PARENT_ID"] == parent_row["SPECIES_CODE"]]
            siblings = siblings[
                ~siblings["SPECIES_CODE"].isna() & (siblings["SPECIES_CODE"] != "")
            ]
            code = parent_row["SPECIES_CODE"] + "_{}".format(len(siblings) + 2)
            df.at[i_row, "SPECIES_CODE"] = code
            any_updated = True

    return df


def get_squidle_lineage(label_scheme_id, subdomain="", strip=".", verbose=0):
    """
    Download a live label scheme from subdomain.squidle.org, and reformat.

    Parameters
    ----------
    label_scheme_id : int
        ID of the label scheme to download.
    subdomain : str, optional
        SQUIDLE subdomain to use, one of `""` or `"soi"`. Default is `""`.
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    pandas.DataFrame
        Dataframe of labels, with columns `"LINEAGE"` and `"NAME"`.
    """
    base_url = utils.build_base_url(subdomain)
    if subdomain == "":
        resource_fmt = "label_scheme/{}/labels"
        labels_key = "labels"
    else:
        resource_fmt = "tag_scheme/{}/nested"
        labels_key = "tag_groups"
    resource = resource_fmt.format(label_scheme_id)
    url = base_url + "/api/" + resource
    result = requests.get(url).json()
    if verbose >= 1:
        print(result["name"])
        print(result["description"])
    return pd.DataFrame.from_records(
        build_squidle_lineage(result[labels_key], strip=strip)
    )


def build_extended_catami(verbose=0):
    """
    Build a unified, extended CATAMI label scheme.

    Combines CATAMI v1.4 with SQUIDLE and Australian Morphospecies Catalogue
    schemes.

    Parameters
    ----------
    verbose : int, optional
        Level of verbosity. Default is `0`.

    Returns
    -------
    pandas.DataFrame
        The unified label scheme.
    """
    # Get CATAMI v1.4
    if verbose >= 1:
        print("Downloading CATAMI v1.4 scheme file from squidle.org")
    df_catami = get_catami_scheme()
    if verbose >= 2:
        print("  CATAMI v1.4 scheme file has {} labels".format(len(df_catami)))
    catami_add_lineage(df_catami, inplace=True)

    # Get the SQUIDLE+ scheme
    if verbose >= 1:
        print("Downloading SQUIDLE+ scheme file")
    df_sqdx = get_squidle_extended_scheme()
    if "SOURCE" not in df_sqdx.columns:
        df_sqdx["SOURCE"] = "SQUIDLE+"
    if verbose >= 2:
        print("  SQUIDLE+ scheme file has {} labels".format(len(df_sqdx)))
    # Change SQUIDLE's "Unscorable." label so it will be coincident with the
    # Australian Morphospecies Catalogue's field "Unscorable".
    df_sqdx["CATAMI_DISPLAY_NAME"] = df_sqdx["CATAMI_DISPLAY_NAME"].str.replace(
        "Unscorable.", "Unscorable"
    )
    catami_add_lineage(df_sqdx, inplace=True)

    # Get the Australian Morphospecies Catalogue scheme
    if verbose >= 1:
        print("Downloading Australian Morphospecies Catalogue scheme file")
    df_amc = get_amc_scheme()
    df_amc["SOURCE"] = "AMC"
    if verbose >= 2:
        print(
            "  Australian Morphospecies Catalogue scheme file has {} labels"
            "".format(len(df_amc))
        )
    catami_add_lineage(df_amc, inplace=True)

    if verbose >= 1:
        print("Merging scheme files")

    # Merge the schemes together
    df_full = pd.concat([df_catami, df_sqdx, df_amc], ignore_index=True)

    # Add in missing LINEAGE and missing CATAMI_DISPLAY_NAME values
    if verbose >= 2:
        print(
            "  Converting between LINEAGE and CATAMI_DISPLAY_NAME to fill"
            " missing values"
        )
    lineage2catami_name(df_full, inplace=True)
    catami_name2lineage(df_full, inplace=True)
    # Add NAME fields
    if verbose >= 2:
        print("  Adding NAME field")
    catami_add_lastnames(df_full, inplace=True)
    # Remove duplicates, keeping the first entry. Ignore casing and periods
    # when detecting duplicates
    df_full["lower_lineage"] = (
        df_full["LINEAGE"].str.strip().str.replace(".", "").str.lower()
    )
    df_full.drop_duplicates(subset="lower_lineage", inplace=True)
    df_full.drop(columns="lower_lineage", inplace=True)
    if verbose >= 2:
        print("  Merged scheme files have {} unique labels".format(len(df_full)))

    # Make sure our scheme is 100% complete by adding in records for the label
    # schemes which are currently live on squidle.org.
    if verbose >= 1:
        print("Downloading live SQUIDLE 1.0 labels")
    df_sqd1 = get_squidle_lineage(1)
    df_sqd1["SOURCE"] = "SQUIDLE 1.0"
    if verbose >= 2:
        print("  There are {} live SQUIDLE 1.0 labels".format(len(df_sqd1)))

    if verbose >= 1:
        print("Downloading live SQUIDLE+ labels")
    df_sqdx = get_squidle_lineage(5, subdomain="soi")
    df_sqdx["SOURCE"] = "SQUIDLE+"
    if verbose >= 2:
        print(
            "  There are {} live SQUIDLE+ labels on https://soi.squidle.org"
            "".format(len(df_sqdx))
        )

    # The live AMC scheme is lazy loaded, so we don't get the whole thing
    # unfortunately
    if verbose >= 1:
        print("Downloading live Australian Morphospecies Catalogue labels")
    df_amc = get_squidle_lineage(7)
    df_amc["SOURCE"] = "AMC"
    if verbose >= 2:
        print(
            "  There are {} live Australian Morphospecies Catalogue labels"
            "".format(len(df_amc))
        )

    if verbose >= 1:
        print("Merging live labels with scheme files")
    df_full = pd.concat([df_full, df_sqd1, df_sqdx, df_amc], ignore_index=True)

    # Add in missing LINEAGE and missing CATAMI_DISPLAY_NAME values
    if verbose >= 2:
        print(
            "  Converting between LINEAGE and CATAMI_DISPLAY_NAME to fill"
            " missing values"
        )
    lineage2catami_name(df_full, inplace=True)
    catami_name2lineage(df_full, inplace=True)

    # Fix typos in LINEAGE
    df_full["LINEAGE"] = df_full["LINEAGE"].str.strip().str.replace("  ", " ")
    df_full["LINEAGE"] = df_full["LINEAGE"].str.replace("Psammacora", "Psammocora")
    df_full["LINEAGE"] = df_full["LINEAGE"].str.replace(
        " Coral-killing sponge", " Coral-killing"
    )
    # Remove duplicates, keeping the first entry. Ignore casing and periods
    # when detecting duplicates
    df_full["lower_lineage"] = (
        df_full["LINEAGE"].str.replace(".", "").str.replace("  ", " ").str.lower()
    )

    df_full.drop_duplicates(subset="lower_lineage", inplace=True)
    df_full.drop(columns="lower_lineage", inplace=True)
    # Add missing parent codes, and create new codes when there are none
    add_missing_codes(df_full, inplace=True, verbose=verbose - 1)
    # Clean up extra white space
    for key in ["CATAMI_DISPLAY_NAME", "CPC_CODES", "NAME"]:
        df_full[key] = df_full[key].str.strip().str.replace("  ", " ")
        df_full[key] = df_full[key].str.replace("Psammacora", "Psammocora")
        df_full[key] = df_full[key].str.replace(
            " Coral-killing sponge", " Coral-killing"
        )

    if verbose >= 2:
        print("  Merged schemes have {} unique labels".format(len(df_full)))

    return df_full


def extended_catami_to_csv(destination, *args, verbose=1, **kwargs):
    """
    Build a unified extended CATAMI label scheme and save it to a CSV file.

    Parameters
    ----------
    destination : str
        Name of CSV file to output.
    verbose : int, optional
        Level of verbosity. Default is `1`.

    Returns
    -------
    pandas.DataFrame
        The unified label scheme.
    """
    t0 = time.time()
    if verbose >= 1:
        print(
            "Will build consolidated CATAMI schema integrated with SQUIDLE"
            " extensions, and save to CSV file {}".format(
                destination,
            ),
            flush=True,
        )

    df = build_extended_catami(*args, verbose=verbose - 1, **kwargs)

    if verbose >= 1:
        print("Saving output to {}".format(destination), flush=True)
    destdir = os.path.dirname(destination)
    if destdir and not os.path.isdir(destdir):
        os.makedirs(destdir, exist_ok=True)
    df.to_csv(destination, index=False)
    if verbose >= 1:
        print(
            "Downloaded, built and saved schema of {} labels in {}".format(
                len(df),
                datetime.timedelta(seconds=time.time() - t0),
            ),
            flush=True,
        )
    return df


def get_parser():
    """
    Build CLI parser for downloading SQUIDLE image dataset.

    Parameters
    ----------
    None

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
        description=(
            "Build an extended CATAMI label scheme, unifying the SQUIDLE+ and"
            " Australian Morphospecies Catalogue schemes which each extend"
            " CATAMI v1.4 in different ways"
        ),
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Output CSV file.",
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

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    extended_catami_to_csv(**kwargs)


if __name__ == "__main__":
    main()
