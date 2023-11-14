import urllib.parse

import ipywidgets
from ipywidgets import GridBox, Layout
from IPython.display import display, HTML, Image

import astropy.coordinates as coord
from astropy import units as u
import numpy as np
import pandas as pd
from memoization import cached

from matplotlib import pyplot as plt

import tic_meta
import catalog
import catalog_stats
import user_stats

import gaia_meta
import tesseb_meta

from common import as_nullable_int, left_outer_join_by_column_merge

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
    "SIMBAD_VAR_OTYPES",
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
    "TESSEB_URL",
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
    width="100%",
):
    col1 = ipywidgets.Output(layout=col1_layout)
    col2 = ipywidgets.Output(layout=col2_layout)
    grid_box = GridBox(
        children=[col1, col2],
        layout=Layout(
            width=width,
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


def n_columns(
    contents,
    also_return_outputs=False,
    layout={"border": "1px dotted gray"},
    grid_template_columns=None,
    width="100%",
):
    num_cols = len(contents)
    outputs = [ipywidgets.Output(layout=layout) for i in range(num_cols)]
    if grid_template_columns is None:
        grid_template_columns = " ".join(["auto" for i in range(num_cols)])

    grid_template_areas = " ".join([f"col{i}" for i in range(num_cols)])

    grid_box = GridBox(
        children=outputs,
        layout=Layout(
            width=width,
            grid_template_columns=grid_template_columns,
            grid_template_areas=grid_template_areas,
        ),
    )

    for content, output in zip(contents, outputs):
        if content is not None:
            with output:
                display(content)

    if also_return_outputs:
        return grid_box, outputs
    else:
        return grid_box


def display_summary(min_eb_score):
    summary = catalog_stats.summary(min_eb_score=min_eb_score)
    total_num_users = user_stats.load_total_num_users_from_file()
    summary["Num. of users"] = total_num_users

    # use dataframe to pretty print the result
    df = pd.DataFrame.from_dict(summary, orient="index")
    df.rename(columns={0: ""}, inplace=True)
    styler = df.style.format(
        {
            # separate thousands
            "": "{:,}",
        }
    )

    display(styler)
    return styler


def get_catalog(type="pht_eb"):
    if type == "pht_eb":
        df = catalog.load_pht_eb_candidate_catalog_from_file(add_convenience_columns=True)
        # Create a TESSEB_URL column that can be used to include links
        # to TESSEB entry of a TIC
        df["TESSEB_URL"] = df["tic_id"]
        return df
    elif type == "tic_meta":
        return tic_meta.load_tic_meta_table_from_file()
    elif type == "hr":
        # Use a small sample of the TICs to plot a simple HR plot
        return tic_meta.load_tic_meta_table_from_file()[:500][["ID", "Teff", "lum"]]
    else:
        raise ValueError(f"Unsupported catalog type: {type}")


def join_pht_eb_candidate_catalog_with(aux_catalogs, df_pht=None):
    """Join PHT EB catalog with a list of others given in ``aux_catalogs``."""

    if df_pht is None:
        df_pht = get_catalog(type="pht_eb")

    for aux_cat in aux_catalogs:
        if aux_cat == "tesseb":
            df_aux = tesseb_meta.load_tesseb_meta_table_from_file(add_convenience_columns=True, one_row_per_tic_only=True)
            df_pht = left_outer_join_by_column_merge(
                df_pht, df_aux,
                join_col_main="tic_id", join_col_aux="TIC",
                prefix_aux="TESSEB"
                )
        elif aux_cat == "gaia":
            df_aux = gaia_meta.load_gaia_dr3_meta_table_from_file(add_variable_meta=True)
            df_pht = left_outer_join_by_column_merge(
                df_pht, df_aux,
                join_col_main="tic_id", join_col_aux="TIC_ID",
                prefix_aux="GAIA"
                )
            # Misc. type fixing after join
            as_nullable_int(df_pht, ["GAIA_Source", "GAIA_SolID"])  # some cells is NA, so convert it to nullable integer

            # placeholders for UI to construct links to GAIA
            # Note: we use the column GAIA_DR3Name as the basis, rather than GAIA_Source, because empirically,
            # - the Source in xmatch result is sometimes not reliable (and points to invalid ID)
            # - the one in DR3Name seems to be more reliable.
            df_pht["GAIA_DR3_URL"] = df_pht["GAIA_DR3Name"].str.replace("Gaia DR3 ", "")
            df_pht["GAIA_DR3_VAR_URL"] = df_pht["GAIA_DR3Name"].str.replace("Gaia DR3 ", "")
            df_pht.loc[pd.isna(df_pht["GAIA_Class"]), "GAIA_DR3_VAR_URL"] = np.nan  # only those with variable class gets URL
        else:
            raise ValueError(f"Unsupported axillary catalog: {aux_cat}")

    return df_pht


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

        display(plot_tic_on_hr(tic_id).get_figure())

    return display(grid)


def plot_tic_on_hr(tic_id):
    df_tic = get_catalog("tic_meta")
    row_tic = df_tic[df_tic["ID"] == tic_id].iloc[0]

    df_hr = get_catalog("hr")

    ax = plt.figure().gca()
    ax.scatter(df_hr["Teff"], df_hr["lum"], s=1, c="gray")
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("Luminosity (sun)")

    if np.isfinite(row_tic["Teff"]) and np.isfinite(row_tic["lum"]):
        ax.scatter(row_tic["Teff"], row_tic["lum"], s=128, c="red", marker="X")
    else:
        print(f"WARN Cannot plot TIC {row_tic['ID']} on H-R diagram. Missing  Teff and/or Luminosity. "
              f"""Teff={row_tic["Teff"]}, lum={row_tic["lum"]}"""
              )
    ax.set_title(f"TIC {row_tic['ID']}")

    return ax


def style(df_catalog, show_thumbnail=False):
    def make_clickable(val, url_template, target, quote_val=False, link_text_func=None):
        if pd.isna(val):
            return val
        val_in_url = str(val)
        if quote_val:
            val_in_url = urllib.parse.quote_plus(val)
        if "%s" not in url_template:
            url = f"{url_template}{val_in_url}"
        else:
            url = url_template.replace("%s", val_in_url)
        link_text = val
        if link_text_func is not None:
            link_text = link_text_func(val)
        return f'<a target="{target}" href="{url}">{link_text}</a>'

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

    def make_gaia_source_clickable_to_gaia_dr3(val):
        gaia_dr3_url_base = "https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ63b1ee071c8561&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=%2BI%2F355%2Fgaiadr3%2BI%2F355%2Fparamp&%2F%2Ftables=I%2F355%2Fgaiadr3&%2F%2Ftables=I%2F355%2Fparamp&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&-out.add=_r&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-oc.form=sexa&-out.src=I%2F355%2Fgaiadr3%2CI%2F355%2Fparamp&-nav=cat%3AI%2F355%26tab%3A%7BI%2F355%2Fgaiadr3%7D%26tab%3A%7BI%2F355%2Fparamp%7D%26key%3Asource%3D%2BI%2F355%2Fgaiadr3%2BI%2F355%2Fparamp%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=+15&-c.u=arcsec&-c.geom=r&-source=&-x.rs=10&-source=I%2F355%2Fgaiadr3+I%2F355%2Fparamp&-out.orig=standard&-out=RA_ICRS&-out=DE_ICRS&-out=Source&-out=Plx&-out=PM&-out=pmRA&-out=pmDE&-out=sepsi&-out=RUWE&-out=Dup&-out=Gmag&-out=BPmag&-out=RPmag&-out=BP-RP&-out=RV&-out=e_RV&-out=Vbroad&-out=GRVSmag&-out=VarFlag&-out=NSS&-out=XPcont&-out=XPsamp&-out=RVS&-out=EpochPh&-out=EpochRV&-out=MCMCGSP&-out=MCMCMSC&-out=Teff&-out=logg&-out=%5BFe%2FH%5D&-out=Dist&-out=A0&-out=HIP&-out=PS1&-out=SDSS13&-out=SKYM2&-out=TYC2&-out=URAT1&-out=AllWISE&-out=APASS9&-out=GSC23&-out=RAVE5&-out=2MASS&-out=RAVE6&-out=RAJ2000&-out=DEJ2000&-out=Pstar&-out=PWD&-out=Pbin&-out=ABP&-out=ARP&-out=GMAG&-out=Rad&-out=Rad-Flame&-out=Lum-Flame&-out=Mass-Flame&-out=Age-Flame&-out=z-Flame&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET&Source="
        return make_clickable(val, gaia_dr3_url_base, "_gaiadr3")

    def make_gaia_source_clickable_to_gaia_dr3_var(val):
        gaia_dr3_var_url_base = "https://vizier.cfa.harvard.edu/viz-bin/VizieR-4?-ref=VIZ63b1f3211239e&-to=-4b&-from=-4&-this=-4&%2F%2Fsource=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fvarisum&%2F%2Ftables=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fveb&%2F%2Ftables=I%2F358%2Fvst&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=I%2F358%2Fvarisum%2CI%2F358%2Fvclassre%2CI%2F358%2Fveb%2CI%2F358%2Fvst&-nav=cat%3AI%2F358%26tab%3A%7BI%2F358%2Fvarisum%7D%26tab%3A%7BI%2F358%2Fvclassre%7D%26tab%3A%7BI%2F358%2Fveb%7D%26tab%3A%7BI%2F358%2Fvst%7D%26key%3Asource%3DI%2F358%2Fvclassre%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-x.rs=10&-source=I%2F358%2Fvarisum+I%2F358%2Fvclassre+I%2F358%2Fveb+I%2F358%2Fvst&-out.orig=standard&-out=Source&-out=RA_ICRS&-out=DE_ICRS&-out=TimeG&-out=DurG&-out=Gmagmean&-out=TimeBP&-out=DurBP&-out=BPmagmean&-out=TimeRP&-out=DurRP&-out=RPmagmean&-out=VCR&-out=VRRLyr&-out=VCep&-out=VPN&-out=VST&-out=VLPV&-out=VEB&-out=VRM&-out=VMSO&-out=VAGN&-out=Vmicro&-out=VCC&-out=SolID&-out=Classifier&-out=Class&-out=_RA.icrs&-out=_DE.icrs&-out=Rank&-out=TimeRef&-out=Freq&-out=magModRef&-out=PhaseGauss1&-out=sigPhaseGauss1&-out=DepthGauss1&-out=PhaseGauss2&-out=sigPhaseGauss2&-out=DepthGauss2&-out=AmpCHP&-out=PhaseCHP&-out=ModelType&-out=Nparam&-out=rchi2&-out=PhaseE1&-out=DurE1&-out=DepthE1&-out=PhaseE2&-out=DurE2&-out=DepthE2&-out=Ampl&-out=NfoVTrans&-out=FoVAbbemean&-out=NTimeScale&-out=TimeScale&-out=Variogram&-meta.ucd=0&-meta=0&-usenav=1&-bmark=GET&Source="
        return make_clickable(val, gaia_dr3_var_url_base, "_gaiadr3_var")

    def make_subject_img_id_image(val):
        # set explicit with here to make it thumbnail. needed to make it work with colab UI
        return f'<img src="https://panoptes-uploads.zooniverse.org/subject_location/{val}.png" style="width: 120px;">'

    def make_tesseb_link(val):
        # TESSEB_URL column is populated with tic id so as to construct the links
        # as follows
        return make_clickable(
            val, "http://tessebs.villanova.edu/search_results?tic=", "_tesseb", link_text_func=lambda val: "details"
        )

    def make_tic_id_clickable_to_vetting_doc(val):
        # Vetting_URL column is populated with tic id so as to construct links to vetting notebook
        return make_clickable(
            val, "https://colab.research.google.com/github/orionlee/pht_eb_stats2/blob/main/vetting/EB_V_%s.ipynb", "_tesseb",
            link_text_func=lambda val: "vetting doc",
        )

    format_spec = {
        "tic_id": make_tic_id_clickable,
        "TIC": make_tic_id_clickable,  # column name used in vetting_statuses.csv
        "best_subject_id": make_subject_id_clickable,
        "SIMBAD_MAIN_ID": make_simbad_id_clickable,
        "ASASSN_URL": make_asas_sn_url_clickable,
        "VSX_OID": make_vsx_id_clickable,
        "TESSEB_URL": make_tesseb_link,
        "GAIA_DR3_URL": make_gaia_source_clickable_to_gaia_dr3,
        "GAIA_DR3_VAR_URL": make_gaia_source_clickable_to_gaia_dr3_var,
        "Vetting_URL": make_tic_id_clickable_to_vetting_doc,
    }

    if show_thumbnail:
        format_spec["best_subject_img_id"] = make_subject_img_id_image

    return df_catalog.style.format(format_spec).hide(axis="index")


def plot_skymap(
    df, ra_colname="TIC_ra", dec_colname="TIC_dec", ax=None, figsize=(12, 9), projection="mollweide", scatter_kwargs=None
):
    # adapted from: https://learn.astropy.org/tutorials/plot-catalog.html

    if scatter_kwargs is None:
        scatter_kwargs = dict()
        scatter_kwargs["s"] = 0.2
        scatter_kwargs["c"] = "gray"
        scatter_kwargs["alpha"] = 0.5

    fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111, projection=projection)
    ra = coord.Angle(df[ra_colname].fillna(np.nan) * u.degree)
    ra = ra.wrap_at(180 * u.degree)
    dec = coord.Angle(df[dec_colname].fillna(np.nan) * u.degree)
    ax.scatter(ra.radian, dec.radian, **scatter_kwargs)
    ax.set_xticklabels(["14h", "16h", "18h", "20h", "22h", "0h", "2h", "4h", "6h", "8h", "10h"])
    ax.grid(True)

    return ax


#
# User-level statistics visualization
#


def plot_cumulative_user_contributions(stats: pd.DataFrame, ranks_to_label=[1, 5, 50, 100], also_return_label_df=False):
    """Plot the (top) users contributions in the form of cumulative percentages of
    subjects covered by the users.

    See `user_stats.load_top_users_contributions_from_table()` :
    load the precomputed cumulative top users contributions that can be plotted
    by this function.
    """
    stats = stats.set_index("rank", drop=True, inplace=False)
    ax = stats.plot(y="cum_num_subjects %", kind="line")
    cum_pct_list = []
    for rank in ranks_to_label:
        cum_pct = stats.loc[rank, "cum_num_subjects %"]
        cum_pct_list.append(cum_pct)
        ax.scatter(rank, cum_pct, marker="X", s=64, color="red")
        ax.annotate(f"rank {rank},\n{cum_pct:.0%}", (rank, cum_pct), xytext=(rank - 4, cum_pct - 0.1), fontsize=12)
    # change x/ y limit to accommodate the annotation
    ax.set_ylim(ax.get_ylim()[0] - 0.1, None)
    ax.set_xlim(None, ax.get_xlim()[1] + 15)

    num_subjects = int(stats.iloc[0]["cum_num_subjects"] / stats.iloc[0]["cum_num_subjects %"])
    ax.set_xlabel("Contributor rank")
    ax.set_ylabel("Cumulative % of tagged subjects")
    ax.set_title(
        f"""Cumulative % of subjects tagged by top contributors
num. of subjects: {num_subjects}
"""
    )
    ax.legend().remove()

    if not also_return_label_df:
        return ax
    else:
        df = pd.DataFrame({"Rank": ranks_to_label, "Cumulative Percentage": cum_pct_list})
        df.set_index("Rank", inplace=True)
        return ax, df
