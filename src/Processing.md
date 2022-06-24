# Processing to get the data

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
python pht_subj_comments_per_comment.py
```

#### Process comments to produce per-subject summary (tags, etc.)

Output: `cache/comments_summary_per_subject.csv`
TBD.
Summary

- Count up-votes (EB-like tags), down votes (transits, NEB, etc.), and score it

Possible useful later on metadata

- num. comments
- last updated


#### Process comments to produce per-TIC summary (tags, etc.)

Output: `cache/comments_summary_per_tic.csv`
TBD


## SIMBAD data

Output: `data/simbad_meta.csv`

Codes in `simbad_meta.py`. It performs the following steps:

### 1. Grab SIMBAD data for those that can be looked up by TIC IDs.

Done by `get_and_save_simbad_meta_of_all_by_tics()`.
Output: `cache/simbad_meta_by_ticid.csv`

TODO: Consider to validate the result of lookup by TIC IDs (rather than just trusting them)

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

Done by `TBD`
Output: `data/simbad_meta.csv`

