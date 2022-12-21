#
# Misc. helper for EB Vetting notebooks
#

from urllib.parse import quote_plus
from IPython.display import display, Markdown
import pandas as pd


def display_no_index(df):
    return display(df.style.hide(axis="index"))


def display_all_meta_highlights(all_meta):
    """Display highlights from matched catalogs of a TIC, primarily on variability."""
    tic = all_meta.pht_eb["tic_id"].iloc[0]

    simbad_header = "### SIMBAD"
    if len(all_meta.simbad) > 0:
        simbad_id = all_meta.simbad["MAIN_ID"].iloc[0]
        simbad_header = (f"{simbad_header}"
                        f"&emsp;[live](https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote_plus(simbad_id)})"
                        f" , [references](https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote_plus(simbad_id)}#lab_bib)")
    display(Markdown(simbad_header))
    display_no_index(all_meta.simbad[["MAIN_ID",  "FLUX_V", "OTYPES", "V__vartyp",]])

    vsx_header = "### VSX"
    if len(all_meta.vsx) > 0:
        vsx_oid = all_meta.vsx["OID"].iloc[0]
        vsx_header = f"{vsx_header}&emsp;[live](https://www.aavso.org/vsx/index.php?view=detail.top&oid={vsx_oid}) "
    display(Markdown(vsx_header))
    display_no_index(all_meta.vsx[["Name",  "Type", ]])

    asas_sn_header = "### ASAS-SN"
    if len(all_meta.asas_sn) > 0:
        asas_sn_obj_url = all_meta.asas_sn["URL"].iloc[0]
        asas_sn_header = f"{asas_sn_header}&emsp;[live]({asas_sn_obj_url})"
    display(Markdown(asas_sn_header))
    display_no_index(all_meta.asas_sn[["ASASSN-V",  "Vmag", "Type", "Prob", "Per"]])


    tesseb_header = f"### TESS EB&emsp;[live](http://tessebs.villanova.edu/{str(tic).zfill(10)})"
    display(Markdown(tesseb_header))
    display_no_index(all_meta.tesseb[["TIC", "Per", "Epochp", "Epochs-pf", "Morph", "Sectors",]])

    gaia_header = "### Gaia DR3 / DR3 Variable"
    if len(all_meta.gaia) > 0:
        gaia_id = all_meta.gaia["Source"].iloc[0]
        gaia_header = f"{gaia_header}&emsp;[live](https://vizier.cds.unistra.fr/viz-bin/VizieR-S?Gaia%20DR3%20{gaia_id})"
        if not pd.isna(all_meta.gaia["Class"].iloc[0]):
            gaia_dr3_var_url = f"https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ63a395c22729a4&-to=-4b&-from=-4&-this=-4&%2F%2Fsource=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fvarisum&%2F%2Ftables=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fveb&%2F%2Ftables=I%2F358%2Fvst&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-nav=cat%3AI%2F358%26tab%3A%7BI%2F358%2Fvarisum%7D%26tab%3A%7BI%2F358%2Fvclassre%7D%26tab%3A%7BI%2F358%2Fveb%7D%26tab%3A%7BI%2F358%2Fvst%7D%26key%3Asource%3DI%2F358%2Fvclassre%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=+I%2F358%2Fvarisum+I%2F358%2Fvclassre+I%2F358%2Fveb+I%2F358%2Fvst&-out.src=I%2F358%2Fvclassre&-out.orig=standard&-out=Source&-out=SolID&-out=Classifier&-out=Class&-out=ClassSc&-out=RA_ICRS&-out=DE_ICRS&-out=_RA.icrs&-out=_DE.icrs&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET&Source={gaia_id}"
            gaia_header = f"{gaia_header} , [variable]({gaia_dr3_var_url})"
    display(Markdown(gaia_header))
    display_no_index(all_meta.gaia[["Source", "RAdeg", "DEdeg", "Gmag",  "BP-RP", "Teff", "RUWE", "sepsi", "Dup", "Class", "ClassSc",]])

