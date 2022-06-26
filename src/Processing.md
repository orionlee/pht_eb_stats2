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
# in case the file exists
rm cache/comments_summary_per_comment.csv
python pht_subj_comments_per_comment.py
```

#### Process comments to produce per-subject summary (tags, etc.)

It requires a mapping of synonyms of EB / transit tags, stored in `../data/pht_tag_map.csv`

Output: `../data/pht_subj_comments_summary.csv `

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

### 1. Grab SIMBAD data for those that can be looked up by TIC IDs.

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

Done by `map_and_save_simbad_otypes_of_all()`
Output: `../data/simbad_is_eb.csv`

Note: the mapping between SIMBAD `OTYPE` to `Is_EB` is driven by `../data/simbad_typemap.csv`
mapping table. The mapping needs to be updated if an `OTYPE` value is not included there.

#### SIMBAD Result summary

- Input: 12561 TICs  (for PHT subjects from sector 1 - 39)
- Output:
  - 11192 TICs matched with high certainty
  - 1369 TICs not matched. Out of these
    - 925 TICs with some SIMBAD records, but are deemed to be likely false matches
    - 444 TIcs with no SIMBAD records
