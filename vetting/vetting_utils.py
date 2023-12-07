#
# Misc. helper for EB Vetting notebooks
#

from types import SimpleNamespace
from urllib.parse import quote_plus
from IPython.display import display, Markdown
import pandas as pd


def display_no_index(df):
    return display(df.style.hide(axis="index"))


def get_tesseb_live_url_of_tic(tic):
    return f"http://tessebs.villanova.edu/{str(tic).zfill(10)}"


def display_all_meta_highlights(all_meta):
    """Display highlights from matched catalogs of a TIC, primarily on variability."""
    tic = all_meta.pht_eb["tic_id"].iloc[0]

    simbad_header = "### SIMBAD"
    if len(all_meta.simbad) > 0:
        simbad_id = all_meta.simbad["MAIN_ID"].iloc[0]
        simbad_header = (
            f"{simbad_header}"
            f"&emsp;[live](https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote_plus(simbad_id)})"
            f" , [references](https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote_plus(simbad_id)}#lab_bib)"
        )
    display(Markdown(simbad_header))
    display_no_index(
        all_meta.simbad[
            [
                "MAIN_ID",
                "FLUX_V",
                "OTYPES",
                "V__vartyp",
            ]
        ]
    )

    vsx_header = "### VSX"
    if len(all_meta.vsx) > 0:
        vsx_oid = all_meta.vsx["OID"].iloc[0]
        vsx_header = f"{vsx_header}&emsp;[live](https://www.aavso.org/vsx/index.php?view=detail.top&oid={vsx_oid}) "
    display(Markdown(vsx_header))
    display_no_index(
        all_meta.vsx[
            [
                "Name",
                "Type",
            ]
        ]
    )

    asas_sn_header = "### ASAS-SN"
    if len(all_meta.asas_sn) > 0:
        asas_sn_obj_url = all_meta.asas_sn["URL"].iloc[0]
        asas_sn_header = f"{asas_sn_header}&emsp;[live]({asas_sn_obj_url})"
    display(Markdown(asas_sn_header))
    display_no_index(all_meta.asas_sn[["ASASSN-V", "Vmag", "Type", "Prob", "Per"]])

    # for TESS EB, link to live TESS EB is always shown, in case new entries have been added.
    tesseb_header = f"### TESS EB&emsp;[live]({get_tesseb_live_url_of_tic(tic)})"
    display(Markdown(tesseb_header))
    display_no_index(
        all_meta.tesseb[
            [
                "TIC",
                "Per",
                "Epochp",
                "Epochs-pf",
                "Morph",
                "Sectors",
            ]
        ]
    )

    gaia_header = "### Gaia DR3 / DR3 Variable"
    if len(all_meta.gaia) > 0:
        gaia_id = all_meta.gaia["Source"].iloc[0]
        # The long URL includes both Gaia DR3 main and astrophysical parameters, with custom set of columns.
        # It replaced the simple
        #   f"https://vizier.cds.unistra.fr/viz-bin/VizieR-S?Gaia%20DR3%20{gaia_id}"
        gaia_dr3_url = f"https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ6553efc5ee22b&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=%2BI%2F355%2Fgaiadr3%2BI%2F355%2Fparamp&%2F%2Fc=07%3A43%3A56.28+-60%3A23%3A47&%2F%2Ftables=I%2F355%2Fgaiadr3&%2F%2Ftables=I%2F355%2Fparamp&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&-out.add=_r&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-oc.form=sexa&-nav=cat%3AI%2F355%26tab%3A%7BI%2F355%2Fgaiadr3%7D%26tab%3A%7BI%2F355%2Fparamp%7D%26key%3Asource%3D%2BI%2F355%2Fgaiadr3%2BI%2F355%2Fparamp%26key%3Ac%3D07%3A43%3A56.28+-60%3A23%3A47%26pos%3A07%3A43%3A56.28+-60%3A23%3A47%28+15+arcsec%29%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=+15&-c.u=arcsec&-c.geom=r&-source=&-out.src=I%2F355%2Fgaiadr3%2CI%2F355%2Fparamp&-x.rs=10&-source=I%2F355%2Fgaiadr3+I%2F355%2Fparamp&-out.orig=standard&-out=RA_ICRS&-out=DE_ICRS&-out=Source&Source={gaia_id}&-out=Plx&-out=PM&-out=pmRA&-out=pmDE&-out=sepsi&-out=RUWE&-out=Dup&-out=Gmag&-out=BPmag&-out=RPmag&-out=BP-RP&-out=RV&-out=e_RV&-out=Vbroad&-out=GRVSmag&-out=VarFlag&-out=NSS&-out=XPcont&-out=XPsamp&-out=RVS&-out=EpochPh&-out=EpochRV&-out=MCMCGSP&-out=MCMCMSC&-out=Teff&-out=logg&-out=%5BFe%2FH%5D&-out=Dist&-out=A0&-out=HIP&-out=PS1&-out=SDSS13&-out=SKYM2&-out=TYC2&-out=URAT1&-out=AllWISE&-out=APASS9&-out=GSC23&-out=RAVE5&-out=2MASS&-out=RAVE6&-out=RAJ2000&-out=DEJ2000&-out=Pstar&-out=PWD&-out=Pbin&-out=ABP&-out=ARP&-out=GMAG&-out=Rad&-out=SpType-ELS&-out=Rad-Flame&-out=Lum-Flame&-out=Mass-Flame&-out=Age-Flame&-out=Flags-Flame&-out=Evol&-out=z-Flame&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET"
        gaia_header = f"{gaia_header}&emsp;[live]({gaia_dr3_url})"
        if not pd.isna(all_meta.gaia["Class"].iloc[0]):
            gaia_dr3_var_url = f"https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ63a395c22729a4&-to=-4b&-from=-4&-this=-4&%2F%2Fsource=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fvarisum&%2F%2Ftables=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fveb&%2F%2Ftables=I%2F358%2Fvst&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-nav=cat%3AI%2F358%26tab%3A%7BI%2F358%2Fvarisum%7D%26tab%3A%7BI%2F358%2Fvclassre%7D%26tab%3A%7BI%2F358%2Fveb%7D%26tab%3A%7BI%2F358%2Fvst%7D%26key%3Asource%3DI%2F358%2Fvclassre%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=+I%2F358%2Fvarisum+I%2F358%2Fvclassre+I%2F358%2Fveb+I%2F358%2Fvst&-out.src=I%2F358%2Fvclassre&-out.orig=standard&-out=Source&-out=SolID&-out=Classifier&-out=Class&-out=ClassSc&-out=RA_ICRS&-out=DE_ICRS&-out=_RA.icrs&-out=_DE.icrs&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET&Source={gaia_id}"
            gaia_header = f"{gaia_header} , [variable]({gaia_dr3_var_url})"
    display(Markdown(gaia_header))
    display(  # cannot use display_no_index(), because somehow the Source value would be corrupted with it.
        all_meta.gaia[
            [
                "Source",
                "RAJ2000",
                "DEJ2000",
                "Gmag",
                "BP-RP",
                "Teff",
                "RUWE",
                "sepsi",
                "e_RV",
                "Dup",
                "Class",
                "ClassSc",
                # column "NSS" would be helpful too but it is not in downloaded meta
            ]
        ]
    )
    display(
        Markdown(
            f"""
### Misc. EB Catalogs from TESS observations
- [TESS OBA-type eclipsing binaries (IJspeert+, 2021)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ655788c4386ef3&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=J%2FA%2BA%2F652%2FA120%2Feb-cat&%2F%2Ftables=J%2FA%2BA%2F652%2FA120%2Feb-cat&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=J%2FA%2BA%2F652%2FA120%2Feb-cat&-nav=cat%3AJ%2FA%2BA%2F652%2FA120%26tab%3A%7BJ%2FA%2BA%2F652%2FA120%2Feb-cat%7D%26key%3Asource%3DJ%2FA%2BA%2F652%2FA120%2Feb-cat%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=J%2FA%2BA%2F652%2FA120%2Feb-cat&-out=TIC&TIC={tic}&-out=RAJ2000&-out=DEJ2000&-out=Jmag&-out=Hmag&-out=Kmag&-out=Gmag&-out=Tmag&-out=LumClass&-out=BPmag&-out=RPmag&-out=WDFlag&-out=SimbadOtype&-out=SimbadSptype&-out=tsupcon&-out=eclPer&-out=eclScore&-out=DepthP&-out=DepthS&-out=WidthP&-out=WidthS&-out=tsupconT&-out=eclPerT&-out=FlagS&-out=FlagVar&-out=FlagCont&-out=FlagHeart&-out=dupl-group&-out=Simbad&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET)
- [TESS Close Binaries in the southern hemisphere (Justesen+, 2021)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ655789fb3aa4f3&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=J%2FApJ%2F912%2F123&%2F%2Ftables=J%2FApJ%2F912%2F123%2Ftable2&%2F%2Ftables=J%2FApJ%2F912%2F123%2Ftable3&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=J%2FApJ%2F912%2F123%2Ftable2%2CJ%2FApJ%2F912%2F123%2Ftable3&-nav=cat%3AJ%2FApJ%2F912%2F123%26tab%3A%7BJ%2FApJ%2F912%2F123%2Ftable2%7D%26tab%3A%7BJ%2FApJ%2F912%2F123%2Ftable3%7D%26key%3Asource%3DJ%2FApJ%2F912%2F123%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=J%2FApJ%2F912%2F123%2Ftable2+J%2FApJ%2F912%2F123%2Ftable3&-out=TIC&TIC={tic}&-out=Per&-out=t1&-out=t2&-out=ecosw&-out=d1&-out=d2&-out=Tmag&-out=Simbad&-out=_RA&-out=_DE&-out=rp&-out=a%2FR1&-out=esinw&-out=inc&-out=fp&-out=Teff1&-out=Teff2&-out=f_Teff&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET)
- [δ Scuti Pulsators in EBs from TESS (Chen+, 2022)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ65578b153bda7e&-to=-4b&-from=-2&-this=-4&%2F%2Fsource=J%2FApJS%2F263%2F34&%2F%2Ftables=J%2FApJS%2F263%2F34%2Ftable1&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=J%2FApJS%2F263%2F34%2Ftable1&-nav=cat%3AJ%2FApJS%2F263%2F34%26tab%3A%7BJ%2FApJS%2F263%2F34%2Ftable1%7D%26key%3Asource%3DJ%2FApJS%2F263%2F34%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=J%2FApJS%2F263%2F34%2Ftable1&-out=TIC&TIC={tic}&-out=RAJ2000&-out=DEJ2000&-out=Tmag&-out=Porb&-out=Pdom&-out=TeffT&-out=loggT&-out=TeffG&-out=loggG&-out=Lbol&-out=EType&-out=PType&-out=NewPB&-out=Ref&-out=S22&-out=P22&-out=Simbad&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET)
- [Pulsating Stars in EA-type eclipsing binaries observed by TESS (Shi+, 2022)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ65578bc23c6846&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=J%2FApJS%2F259%2F50%2Ftable1&%2F%2Ftables=J%2FApJS%2F259%2F50%2Ftable1&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=J%2FApJS%2F259%2F50%2Ftable1&-nav=cat%3AJ%2FApJS%2F259%2F50%26tab%3A%7BJ%2FApJS%2F259%2F50%2Ftable1%7D%26key%3Asource%3DJ%2FApJS%2F259%2F50%2Ftable1%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-source=J%2FApJS%2F259%2F50%2Ftable1&-out=TIC&TIC={tic}&-out=f_TIC&-out=Name&-out=RAJ2000&-out=DEJ2000&-out=Cons&-out=Per&-out=u_Per&-out=mag1&-out=mag2&-out=n_mag&-out=l_mag&-out=Filter&-out=Sep&-out=Simbad&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET)
- [Multiply eclipsing candidates from TESS satellite (Zasche+, 2022)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ65578c393cd155&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=J%2FA%2BA%2F664%2FA96&%2F%2Ftables=J%2FA%2BA%2F664%2FA96%2Ftable1&%2F%2Ftables=J%2FA%2BA%2F664%2FA96%2Ftablea1&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=J%2FA%2BA%2F664%2FA96%2Ftable1%2CJ%2FA%2BA%2F664%2FA96%2Ftablea1&-nav=cat%3AJ%2FA%2BA%2F664%2FA96%26tab%3A%7BJ%2FA%2BA%2F664%2FA96%2Ftable1%7D%26tab%3A%7BJ%2FA%2BA%2F664%2FA96%2Ftablea1%7D%26key%3Asource%3DJ%2FA%2BA%2F664%2FA96%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-x.rs=10&-source=J%2FA%2BA%2F664%2FA96%2Ftable1+J%2FA%2BA%2F664%2FA96%2Ftablea1&-out=RAJ2000&-out=DEJ2000&-out=VSX&-out=n_VSX&-out=TIC&TIC={tic}&-out=magmax&-out=JD0A&-out=Per&-out=DP&-out=n_DP&-out=DS&-out=n_DS&-out=JD0O&-out=Com&-out=SimbadName&-out=EBtype&-out=PerA&-out=DPA&-out=DSA&-out=PerB&-out=DPB&-out=DSB&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET)
"""
        )
    )


def _tce_reference_name_with_year(publication_year):
    return f"TESS Threshold Crossing Event (TCE), {publication_year}, (online data)"


# comonly used references
BIBS = SimpleNamespace(
    TESS_N="Ricker, G. R.; et al., 2014, Transiting Exoplanet Survey Satellite (TESS)",
    TESS_B="2014SPIE.9143E..20R",
    TESS_SPOC_N="Caldwell, D. A.; et al., 2020, TESS Science Processing Operations Center FFI Target List Products",
    TESS_SPOC_B="2020RNAAS...4..201C",
    QLP_N="Huang, C. X.; et al., 2020, Photometry of 10 Million Stars from the First Two Years of TESS Full Frame Images: Part I",
    QLP_B="2020RNAAS...4..204H",
    TCE_N=_tce_reference_name_with_year,
    # links to TCE is case specific
    TIC_N="Stassun, K. G.; et al., 2019, The Revised TESS Input Catalog and Candidate Target List",  # the paper describing TIC v8, the subsequent paper for v8.1/8.2 focuses mainly on the changes and is not as helpful
    TIC_B="2019AJ....158..138S",
    ASAS_SN_N="Kochanek, C. S.; et al., 2017, The All-Sky Automated Survey for Supernovae (ASAS-SN) Light Curve Server v1.0",
    ASAS_SN_B="2017PASP..129j4502K",
    GAIA_DR3_N="Gaia collaboration; et al., 2022, Gaia Data Release 3 (Gaia DR3) Part 1 Main source",
    GAIA_DR3_B="2022yCat.1355....0G",
    GAIA_DR3_ASTROPHY_N="Creevey, O. L.; et al., 2022, Gaia Data Release 3: Astrophysical parameters inference system (Apsis) I -- methods and content overview",
    GAIA_DR3_ASTROPHY_B="2022arXiv220605864C",
    GAIA_DR3_VAR_N="Gaia collaboration; et al., 2022, Gaia Data Release 3 (Gaia DR3) Part 4 Variability",
    GAIA_DR3_VAR_B="2022yCat.1358....0G",
    TESSEB_N="Prša, A.; et al., 2022, TESS Eclipsing Binary Stars. I. Short-cadence Observations of 4584 Eclipsing Binaries in Sectors 1-26",
    TESSEB_B="2022ApJS..258...16P",
    TESSEB_LIVE_N="TESS Eclipsing Binary Catalogue (online data)",
    # links to live TESS EB is case specific
    # PHt II paper discussed methodlogy, with (indirect) mentions of eclipsingbinary tagging
    PHT_II_N="Planet Hunters TESS II: findings from the first two years of TESS",
    PHT_II_B="2021MNRAS.501.4669E",
)
