import urllib.parse

import ipywidgets
from ipywidgets import GridBox, Layout
from IPython.display import display, HTML, Image

import pandas as pd
from memoization import cached

from matplotlib import pyplot as plt

import tic_meta
import catalog

#
# Constants to select commonly used column subsets from various catalogs
#

# For PHT EB catalog
CAT_COLS_COMMON = [
    "tic_id",
    "best_subject_id",
    "is_eb_catalog",
    "eb_score",
    "SIMBAD_MAIN_ID",
    "SIMBAD_OTYPES",
    "SIMBAD_Is_EB",
    "VSX_OID",
    # VSX_Name is more human friendly, but I can use OID and go to the VSX page directly
    # "VSX_Name",
    "VSX_Type",
    "VSX_Is_EB",
    "VSX_Period",
    "ASASSN_Name",
    "ASASSN_URL",
    "ASASSN_Type",
    "ASASSN_Per",
]

# For TIC Meta
TIC_COLS_COMMON = [
    "ID",
    "Tmag",
    "Vmag",
    "contratio",
    "Teff",
    "rad",
    "mass",
    "rho",
    "lumclass",
    "lum",
    "ra",
    "dec",
    "d",
    "plx",
    "pmRA",
    "pmDEC",
    "disposition",
    "duplicate_id",
    "priority",
    "GAIA",
]


def two_columns(
    col1_content=None,
    col2_content=None,
    also_return_outputs=False,
    col1_layout={"border": "1px dotted gray"},
    col2_layout={"border": "1px dotted gray"},
    grid_template_columns="50% 50%",
    grid_template_rows="auto auto",
    width="100%",
):
    col1 = ipywidgets.Output(layout=col1_layout)
    col2 = ipywidgets.Output(layout=col2_layout)
    grid_box = GridBox(
        children=[col1, col2],
        layout=Layout(
            width=width,
            grid_template_rows=grid_template_rows,
            grid_template_columns=grid_template_columns,
            grid_template_areas="""
            "col1 col2"
            """,
        ),
    )

    if col1_content is not None:
        with col1:
            display(col1_content)
    if col2_content is not None:
        with col2:
            display(col2_content)

    if also_return_outputs:
        return grid_box, col1, col2
    else:
        return grid_box


def get_catalog(type="pht_eb"):
    if type == "pht_eb":
        return catalog.load_pht_eb_candidate_catalog_from_file()
    elif type == "tic_meta":
        return tic_meta.load_tic_meta_table_from_file()
    elif type == "hr":
        # Use a small sample of the TICs to plot a simple HR plot
        return tic_meta.load_tic_meta_table_from_file()[:500][["ID", "Teff", "lum"]]
    else:
        raise ValueError(f"Unsupported catalog type: {type}")


_CATALOG_TIC_ID_COLNAME_ = {
    "pht_eb": "tic_id",
    "tic_meta": "ID",
}


def display_details(tic_id, type="tic_meta", brief=True):
    grid, col1, col2 = two_columns(also_return_outputs=True)

    with col1:
        df = get_catalog(type)
        colname_tic_id = _CATALOG_TIC_ID_COLNAME_.get(type)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(
                HTML(
                    f"""
            <a href="https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
            _target="_exofop">TIC {tic_id}</a>"""
                )
            )
            row = df[df[colname_tic_id] == tic_id].iloc[0]
            if brief:
                row = row[TIC_COLS_COMMON]
            display(row.to_frame(name="Value"))

    with col2:
        df_catalog = get_catalog("pht_eb")
        row_catalog = df_catalog[df_catalog["tic_id"] == tic_id].iloc[0]
        subject_id = row_catalog["best_subject_id"]
        display(
            HTML(
                f"""
        <a href="https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess/talk/subjects/{subject_id}"
        _target="_pht_subject">Subject {subject_id}</a>"""
            )
        )
        img_url = f"https://panoptes-uploads.zooniverse.org/subject_location/{row_catalog['best_subject_img_id']}.png"
        display(Image(url=img_url))

        df_tic = get_catalog("tic_meta")
        row_tic = df_tic[df_tic["ID"] == tic_id].iloc[0]
        display(plot_tic_on_hr(row_tic).get_figure())

    return display(grid)


def plot_tic_on_hr(row_tic):
    df_hr = get_catalog("hr")

    ax = plt.figure().gca()
    ax.scatter(df_hr["Teff"], df_hr["lum"], s=1, c="gray")
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("Luminosity (sun)")

    ax.scatter(row_tic["Teff"], row_tic["lum"], s=128, c="red", marker="X")
    ax.set_title(f"TIC {row_tic['ID']}")

    return ax


def style(df_catalog, show_thumbnail=False):
    def make_clickable(val, url_prefix, target, quote_val=False, link_text_func=None):
        if pd.isna(val):
            return val
        val_in_url = val
        if quote_val:
            val_in_url = urllib.parse.quote_plus(str(val))
        link_text = val
        if link_text_func is not None:
            link_text = link_text_func(val)
        return f'<a target="{target}" href="{url_prefix}{val_in_url}">{link_text}</a>'

    def make_tic_id_clickable(val):
        return make_clickable(val, "https://exofop.ipac.caltech.edu/tess/target.php?id=", "_exofop")

    def make_subject_id_clickable(val):
        return make_clickable(
            val, "https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess/talk/subjects/", "_pht"
        )

    def make_simbad_id_clickable(val):
        return make_clickable(val, "https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=", "_simbad")

    def make_vsx_id_clickable(val):
        return make_clickable(val, "https://www.aavso.org/vsx/index.php?view=detail.top&oid=", "_vsx")

    def make_asas_sn_url_clickable(val):
        return make_clickable(val, "", "_asas_sn", link_text_func=lambda val: "details")

    def make_subject_img_id_image(val):
        # Note: setting custom dimension is a bit tricky and is abandoned for now.
        # height can be changed with CSS height easily,
        # but CSS width seems to be constrained by the overall table styling, and cannot make the image wider
        # One probably needs to specify the styles on the column <tds>
        return f'<img src="https://panoptes-uploads.zooniverse.org/subject_location/{val}.png">'

    def abbreviate_simbad_otypes(val):
        # hide common types not useful for analysis
        if pd.isna(val):
            return val
        # remove generic types, also remove duplicates
        return "|".join(set(val.split("|")) - set(["*", "PM*", "IR"]))

    format_spec = {
        "tic_id": make_tic_id_clickable,
        "best_subject_id": make_subject_id_clickable,
        "SIMBAD_MAIN_ID": make_simbad_id_clickable,
        "SIMBAD_OTYPES": abbreviate_simbad_otypes,
        "ASASSN_URL": make_asas_sn_url_clickable,
        "VSX_OID": make_vsx_id_clickable,
    }

    if show_thumbnail:
        format_spec["best_subject_img_id"] = make_subject_img_id_image

    return df_catalog.style.format(format_spec).hide(axis="index")
