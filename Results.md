# Planet Hunters TESS Eclipsing Binary Candidate Catalog

## At a glance

|                                                 |        |
| :---------------------------------------------- | -----: |
| Num. of PHT Subjects                            |  28092 |
| Num. of EBs / EB Candidates (TICs)              |  12561 |
| Num. of EBs / EB Candidates with high certainty |   4066 |
| Estimated num. of new EBs                       |   1480 |
| Sectors                                         | 1 - 39 |
| Num. of users                                   |   1320 |

The candidates are cross-matched against SIMBAD, VSX, and ASAS-SN to

* differentiate between known eclipsing binary and candidates .
* calculate a proxy of tagging accuracy.

Num. of EBs / EB Candidates with high certainty: defined by `eb_score >= 3`, which roughly means at least 3 users have tagged a subject as an eclipsing binary. The subset has a proxy accuracy >= 90%.

See:

* [dashboard notebook](src/dashboard.ipynb) for more detailed breakdown.
* the [catalog in csv](data/catalog_pht_eb_candidates.csv)
* the [pilot study](https://github.com/orionlee/pht_eb_stats/blob/main/PilotResult.md). The current result uses similar methodology, and applying it to the complete data set of sectors 1 - 39.
  * Note: the `eb_score` used here corresponds to `N_eb_adj` used in the pilot study.


### User Contributions

* 1320 users contributed to the tagging.
* The proxy tagging accuracy remains quite stable over the 3 year period: ~90+% at`eb_score >= 3`, or ~80% overall.
  * In fact, there was some decrease in sectors 30 -39
* Top users contributed to majority of the tagging:

|   Rank |   Cumulative Percentage |
|-------:|------------------------:|
|      1 |                39% |
|      5 |                52% |
|     20 |                64% |
|     50 |                73% |
|    100 |                81% |

* Some form of user weighting could be helpful.

See [participants dashboard notebook](src/dashboard_participants.ipynb) for more detailed breakdown.


## Next Steps

* [ ] Vetting of a subset of targets to answer
  * [ ] for those counted as false positives (`is_eb_catalog == "F"`), how many are indeed false positives? Is there some pattern? Some suspicions include:
    * Those listed as RR Lyrae in other catalogs are probably genuine false positives: users probably mistreat them as w-uma
    * Those listed as rotators in other catalogs: some of them possibly have real eclipses in addition to the rotational variability listed, and should be counted as proper match.
  * [ ] for those counted as no data in other catalogs (`is_eb_catalog == "-"`)
    * [ ] what is the tagging accuracy?
    * [ ] A number of classifications are treated as no data as they are deemed to be irrelevant in the context of eclipsing binary, e.g., star in SIMBAD, various eruptive / cataclysmic types. Is such treatment appropriate?
* [ ] Review cross matching (with SIMBAD, etc.) to see if we have included too many invalid ones (false positives) or excluded too many genuine ones (false negatives)
  * [ ] When matching plx / PM, consider to use the error supplied in the catalog to determine if it is a match.
* [ ] Review tag tallying and computation of `eb_score`
  * [ ] Add tags to be counted, in particular, `#EB` and `#E.B.` are employed by some users, but they are not treated as tags in Zooniverse.
    * There are ~3000 such comments, while there are about ~60000 eclipsingbinary comments.
    * other tags to consider: #possibleEB
  * [ ] Handle cases that users tag a subject both as eclipsing binary and transit. (Currently it is treated as a neutral vote)
  * [ ] Consider additional tags counted as dissenting voices. Candidates include `#rr-lyrae` (and possibly other pulsators / rotators)
  * [ ] Should we consider `#contamination` and/or `#NEB` ?
* [x] Cross match with Gaia DR3 for variable status, RUWE, etc.
* [ ] Produce a list of vetted candidates.

## Data Sources / Credits

* [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess/) participants
* The 3 catalogs: [VSX](https://www.aavso.org/vsx/), [ASAS-SN](https://asas-sn.osu.edu/variables), [SIMBAD](http://simbad.u-strasbg.fr/simbad/)
  * Access is done by using [astroquery](https://astroquery.readthedocs.io/) package via [Vizier](https://vizier.u-strasbg.fr/).
* TIC parameters from  [MAST](https://mast.stsci.edu/)
  * TIC parameters access is done by using astroquery.
