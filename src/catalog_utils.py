from types import SimpleNamespace

import catalog
import simbad_meta
import vsx_meta
import asas_sn_meta
import gaia_meta
import tesseb_meta


def get_all_meta_of_tic(tic):
    res = SimpleNamespace()

    df_pht_eb = catalog.load_pht_eb_candidate_catalog_from_file()
    df_pht_eb = df_pht_eb[df_pht_eb["tic_id"] == tic]
    res.pht_eb = df_pht_eb

    df_simbad = simbad_meta.load_simbad_meta_table_from_file()
    df_simbad = df_simbad[df_simbad["TIC_ID"] == tic]
    res.simbad = df_simbad

    df_vsx = vsx_meta.load_vsx_meta_table_from_file()
    df_vsx = df_vsx[df_vsx["TIC_ID"] == tic]
    res.vsx = df_vsx

    df_asas_sn = asas_sn_meta.load_asas_sn_meta_table_from_file()
    df_asas_sn = df_asas_sn[df_asas_sn["TIC_ID"] == tic]
    res.asas_sn = df_asas_sn

    df_gaia = gaia_meta.load_gaia_dr3_meta_table_from_file(add_variable_meta=True)
    df_gaia = df_gaia[df_gaia["TIC_ID"] == tic]
    res.gaia = df_gaia

    df_tesseb = tesseb_meta.load_tesseb_meta_table_from_file(add_convenience_columns=True)
    df_tesseb = df_tesseb[df_tesseb["TIC"] == tic]
    res.tesseb = df_tesseb

    return res
