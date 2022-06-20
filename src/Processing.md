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
TBD

#### Process comments to produce per-subject summary (tags, etc.)

Output: `cache/comments_summary_per_subject.csv`
TBD

#### Process comments to produce per-TIC summary (tags, etc.)

Output: `cache/comments_summary_per_tic.csv`
TBD
