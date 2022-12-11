# Processing to get the data for the catalog

## PHT Subjects with eclipsing binary tag

### Subject IDs

Output: `../data/pht_subj_ids.csv`

- Grab all data from Zooniverse

```shell
python pht_subj_ids.py
```

- Sort the Subject IDs (Zooniverse output is mostly sorted but not)

```shell
cd ../data
cp pht_subj_ids.csv pht_subj_ids_raw.csv
sort -n pht_subj_ids_raw.csv -o pht_subj_ids.csv
```

- Truncate the subject IDs beyond sector 40 or later
  - 68601250 is the first sector 40 subject, as of the time of writing.

- Add `subject_id` header to the csv for convenience


### Subject Metadata (TIC, sector, etc.)

Output: `../data/pht_subj_meta.csv`

```shell
echo subject_id,tic_id,sector,img_id > ../data/pht_subj_meta.csv
python pht_subj_meta.py
```

### Subject Comments

#### Comments raw data (as a staging area)

Output: json files in directory `cache/comments/`

```shell
python pht_subj_comments.py
```

#### Process comments to produce per-comment summary (tags, etc.)

Output: `cache/comments_summary_per_comment.csv`

```shell
# in case the file exists
rm cache/comments_summary_per_comment.csv
python pht_subj_comments_per_comment.py
```

#### Process comments to produce per-subject summary (tags, etc.)

It requires a mapping of synonyms of EB / transit tags, stored in `../data/auxillary/pht_tag_map.csv`

Output: `../data/pht_subj_comments_summary.csv`

```shell
python pht_subj_comments_per_subject.py
```

- Count up-votes (EB-like tags), down votes (transits.), and score it
  - also include some generic metadata, number of comments, last updated, etc.

- TODO:
  - consider other up votes: the text #EB in messages
  - consider EB variants, #algol, #beta-lyr, #w-uma, #contactbinary, etc.
  - consider other down votes, #NEB,  #contamination, pulsators (#rr-lyrae), etc.
Possible useful later on metadata


### Per-TIC PHT Subject Statistics

Output: `../data/tic_pht_stats.csv`

It includes at per-TIC level:

- metadata (relevant PHT subjects)
- summary for comments (EB scores, tags)

```shell
python tic_pht_stats.py
```


## SIMBAD data

Output: `../data/simbad_meta.csv`

Codes in `simbad_meta.py`. It performs the following steps:

### 1. Grab SIMBAD data for those that can be looked up by TIC IDs

Done by `get_and_save_simbad_meta_of_all_by_tics()`.
Output: `cache/simbad_meta_by_ticid.csv`

### 2. Grab SIMBAD data for those that cannot be looked up by TIC IDs

#### 2a. Grab SIMBAD MAIN_IDs using coordinates match via Vizier Crossmatch

Done by `xmatch_and_save_all_unmatched_tics()`
Output: `cache/simbad_tics_xmatch.csv`

#### 2b. Grab SIMBAD data by MAIN_IDs obtained

Done by `get_and_save_simbad_meta_of_all_by_xmatch()`
Output: `cache/simbad_meta_candidates_by_xmatch.csv`

For a MAIN_ID, there could be multiple matches. We lookup and save the closest 5 for further filtering.

#### 2c. For each unmatched TIC, find the closest match from the SIMBAD data obtained

Done by `find_and_save_simbad_best_xmatch_meta()`
Output: `cache/simbad_meta_by_xmatch.csv`

#### 2d. Merge the result from TIC ID lookup with the result from coordinate crossmatch

Produce the final output of SIMBAD metadata for all the TICs.
It also filters out those with low match scores, i.e., likely to be false matches

Done by `combine_and_save_simbad_meta_by_tics_and_xmatch()`
Output: `../data/simbad_meta.csv`

#### 2e. For each SIMBAD record, map `Is_EB`, i.e., if it represents an eclipsing binary

Done by `map_and_save_simbad_is_eb_of_all()`
Output: `../data/simbad_is_eb.csv`

Note: the mapping between SIMBAD `OTYPE` to `Is_EB` is driven by `../data/auxillary/simbad_typemap.csv`
mapping table. The mapping needs to be updated if an `OTYPE` value is not included there.

#### SIMBAD Result summary

- Input: 12561 TICs  (for PHT subjects from sector 1 - 39)
- Output:
  - 11192 TICs matched with high certainty
  - 1369 TICs not matched. Out of these
    - 925 TICs with some SIMBAD records, but are deemed to be likely false matches
    - 444 TIcs with no SIMBAD records


## VSX data

Outputs:

- `../data/vsx_meta.py`
- `../data/vsx_is_eb.csv`

Steps:

1. Get VSX records using Vizier Crossmatch (`xmatch_and_save_vsx_meta_of_all_by_tics()`)
2. Process crossmatch result to find the best match for each TIC (`find_and_save_vsx_best_xmatch_meta()`)
3. Map variable types in the resulting VSX meta to `Is_EB`  (`map_and_save_vsx_is_eb_of_all()`)

Note: the mapping between VSX `TYPE` to `Is_EB` is driven by `../data/auxillary/vsx_vartype_map.csv`
mapping table. The mapping needs to be updated if an `OTYPE` value is not included there.

```shell
python vsx_meta.py
```

Crossmatch best candidate selection TODOs:

- Should CV band map to Gaia instead?, e.g. TIC 8769657
- Consider to accept them if the  mag difference is between 1 and 1.5, and angular distance is close
- review reject tables to see if the rejection is overly aggressive or just about right.
- handle possible duplicates (`df["V"] == 3`)

## ASAS-SN data

Outputs:

- `../data/asas_sn_meta.py`
- `../data/asas_sn_is_eb.csv`

Steps:

1. Get ASAS-SN records using Vizier Crossmatch (`xmatch_and_save_asas_sn_meta_of_all_by_tics()`)
2. Process crossmatch result to find the best match for each TIC (`find_and_save_asas_sn_best_xmatch_meta()`)
3. Map variable types in the resulting ASAS-SN meta to `Is_EB`  (`map_and_save_asas_sn_is_eb_of_all()`)

Note: ASAS-SN variable type is the same as VSX.

```shell
python asas_sn_meta.py
```

## User-level statistics

- `../data/users_top_cum_contributions.csv` : Top users' cumulative contributions
- `../data/tic_eb_rank_groups.csv`: For each TIC, whether it has been tagged by user of specific rank groups. Used to analyze the tagging accuracy of different groups of users.

```shell
python user_stats.py
```


## PHT EB Candidate Catalog

### Create a preliminary catalog

- Output: `../data/catalog_pht_eb_candidates.csv`

- it would contain a fair amount of false positives that require further filtering

- Combine:
  - Per-TIC PHT Subject Statistics
  - SIMBAD results
  - VSX results
  - TIC metadata

- It also creates some user-level statistics. Outputs:
  - `../data/users_top_cum_contributions.csv`

- TODO:
  - for VSX_Is_EB, if the tic has no matching VSX, consider to make it `NA` rather than `-`
  - for cases with `-`, there is still classification, they aren't deemed helpful by initial mapping
  - maybe for SIMBAD, but SIMBAD entries has generic typing like star that is almost certainly not useful


```shell
python catalog.py --combine
```

If there are updates to various mapping tables used or mapping logic,
re-apply all to produce the catalog table.

```shell
python catalog.py --remap
# then re-calculate user-level stats
python user_stats.py
```

---

# Processing to get supplementary data

There is some supplementary data that is not part of the catalog, but is used in subsequent vetting and analysis

Outputs:

- `../data/tesseb_meta.py`

Steps:

1. Get TESSEB records using Vizier (Vizier has a static snapshot for sectors 1 - 26 data)
2. For TICs not found in Vizier, get TESSEB records from Live TESS EB database
   - remove TICs already found in Vizier
   - remove TICs with max of sector tagged in PHT is <= 26 (they should be covered by Vizier if data is real)
   - fetch and scrape Live TESS EB. Consider throttling and caching the HTML result.
3. Combine the two lists.

## TESS EB crossmatch

```shell
python tesseb_meta.py
```
