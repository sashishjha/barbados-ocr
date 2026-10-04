# Data

The competition data is not redistributed here. Download it from the challenge page on Zindi
(R.O.A.D. Barbados Historic Handwriting Challenge) and place it as:

```
data/Train.csv
data/Test.csv
data/SampleSubmission.csv
data/images/<ID>.jpg          # all train + test line crops
```

Then build the folds and cleaned labels (writes `data/derived_checkpoint/`):

```bash
sbatch slurm/01_prepare_data.sbatch
```
