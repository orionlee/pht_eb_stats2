import ipywidgets
from ipywidgets import GridBox, Layout
from IPython.display import display, HTML, Image

import pandas as pd
from memoization import cached

import tic_meta
import catalog


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
    else:
        raise ValueError(f"Unsupported catalog type: {type}")


_CATALOG_TIC_ID_COLNAME_ = {
    "pht_eb": "tic_id",
    "tic_meta": "ID",
}

# TODO: move to catalog_stats.py
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

    return display(grid)
