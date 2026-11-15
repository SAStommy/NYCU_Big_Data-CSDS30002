# Big Data Course - Assignment 2

## Main model and project layout

The final model for this assignment is the **Q4 combined model**. It combines the relationship features selected for Q2 with the temporal features selected for Q3, and uses the RelGNN implementation in the sibling `RelGNN/` folder.

```text
hw2/
├── README.md
├── RelGNN/                              # RelGNN source code and its MIT license
└── BigData2026-Assignment2/
    ├── data/db/                         # Prepared Parquet tables used by the model
    ├── submission/                      # Test input and generated prediction
    ├── q2_ablation.csv                  # Q2 selection/validation results
    ├── q3_ablation.csv                  # Q3 selection/validation results
    ├── q2_selection.txt                 # Selected Q2 variant summary
    ├── q3_selection.txt                 # Selected Q3 variant summary
    └── scripts/                         # Preprocessing, model, experiment, and evaluation code
```

The submitted folder already contains the prepared tables and Q2/Q3 ablation results, so the final model can be run directly without repeating the earlier experiments. The selection text files document the chosen variants. The Q4 runner reads `q2_ablation.csv` and `q3_ablation.csv` to determine the variants to combine.

### Model code

The model uses the RelGNN module in `../RelGNN` (including its atomic-route and message-passing components), adapted to this assignment's eight GitHub activity tables and their primary/foreign-key relationships. `scripts/native_relgnn.py` prepares table-specific features and time-cutoff graph snapshots, then builds the assignment's RelGNN model. `scripts/run_experiment.py` trains the baseline and Q2/Q3 variants, selects feature settings using validation results, trains the combined Q4 model, and writes predictions. The four Q1-Q4 notebooks are also included as question-by-question references.

## Environment

Use Python 3.10. The model uses PyTorch, PyTorch Geometric, PyTorch Frame, RelBench, pandas, NumPy, scikit-learn, and matplotlib. RelGNN's upstream installation notes explain how to choose the PyG wheel compatible with your installed PyTorch/CUDA version: [RelGNN installation instructions](https://github.com/snap-stanford/RelGNN#installation).

From the repository root, create and activate an environment, then install the RelGNN/RelBench dependencies using the upstream instructions. For example:

```powershell
conda create -n relgnn python=3.10 -y
conda activate relgnn
pip install relbench[full]
# Install pyg-lib using the wheel URL matching your PyTorch and CUDA versions.
pip install torch-frame==0.2.3
pip install scikit-learn==1.5.2
pip install pandas numpy matplotlib pyarrow
```

The `RelGNN` checkout must remain next to `BigData2026-Assignment2`, as shown in the project layout. `scripts/native_relgnn.py` imports the model and atomic-route code from `../RelGNN/examples`.

## Run the final Q4 model

Open PowerShell at the repository root and run:

```powershell
cd BigData2026-Assignment2
python scripts/run_experiment.py --part q4
```

If using a virtual environment, replace `python` with its interpreter, for example:

```powershell
.venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part q4
```

Q4 reads the prepared tables from `data/db/`, selects the Q2 and Q3 variants from `q2_ablation.csv` and `q3_ablation.csv`, trains the combined model, and writes the prediction file to:

```text
submission/output_315554001.csv
```

The model needs a compatible GPU for practical runtime; it falls back to CPU when CUDA is unavailable, which may take substantially longer.

### How the runner splits the data

`scripts/run_experiment.py` defines the weeks in UTC; the boundaries correspond to Sunday-to-Sunday weeks in Taiwan time.

- **Training windows:** three earlier labeled weeks (Taiwan time 2026-10-04 to 2026-10-25) train each model.
- **Rolling validation:** two successive weeks (2026-10-18 to 2026-10-25 and 2026-10-25 to 2026-11-01) are predicted using only earlier weeks. The validation predictions are used to choose one decision threshold per activity and to compare Q2/Q3 variants.
- **Fixed historical holdout:** Taiwan time 2026-11-01 to 2026-11-08 is held out when each variant is scored for comparison. Its score is recorded in the ablation results; it is a historical evaluation week, not the assignment's submission test case. Q4 then includes this now-labeled historical week when fitting the final submission model.
- **Final submission test case:** Q4 predicts the user IDs in `submission/test_input.csv` for the assignment's future target period (2026-11-22 to 2026-11-29, Taiwan time). The activity labels for this target period are not available locally. Predictions use the per-activity thresholds chosen on rolling validation. The bundled GH Archive data ends on 2026-11-07, so the snapshot cannot include events from the gap between that collection end and the final prediction cutoff.

The final test CSV contains one row per supplied `UserID` and six binary event predictions. The historical fixed holdout is used to report a score; it does not replace or reveal the future submission labels.

## Re-run all experiments

To regenerate the baseline (Q1), Q2 and Q3 ablations and then train/predict with Q4, run this single command from `BigData2026-Assignment2`:

```powershell
python scripts/run_experiment.py --part all
```

Or run the parts in order:

```powershell
python scripts/run_experiment.py --part baseline
python scripts/run_experiment.py --part q2
python scripts/run_experiment.py --part q3
python scripts/run_experiment.py --part q4
```

`--part all` runs the baseline first, then Q2, Q3, and Q4. Re-running these experiments retrains multiple model variants and therefore takes considerably longer than the direct Q4 command. Q2 and Q3 write their ablation tables, plots, and selection summaries into the project folder; Q4 uses the generated ablation tables to choose its inputs.

### Rebuild the prepared tables (optional)

The submitted `data/db/` folder already contains the eight prepared Parquet tables. To rebuild them from raw GH Archive events, run these commands from `BigData2026-Assignment2`:

```powershell
python scripts/get_gharchive.py
python scripts/preprocess.py --raw raw_data --out data
```

`get_gharchive.py` downloads 24 hourly JSON archives per day for **2026-10-01 through 2026-11-07 inclusive**, into `raw_data/`. It skips files already downloaded. To fetch a different period, edit `start_date` and `end_date` near the top of that script before running it. The preprocessing command consumes those archives, filters and anonymizes the cohort's events, and writes the eight tables to `data/db/`. After rebuilding, run the Q1-Q4 sequence above to regenerate the baseline and Q2/Q3 selection results before running Q4. Raw archives are not included in the submission; they are available from [GH Archive](https://www.gharchive.org/).

## Check the prediction file

From `BigData2026-Assignment2`, check the CSV format and confirm that its user IDs match the provided test list:

```powershell
python scripts/evaluate.py --pred submission/output_315554001.csv --input submission/test_input.csv
```

The prediction CSV must have columns `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`; each activity prediction must be `0` or `1`.

## Data source and licenses

- **RelGNN source code:** The `RelGNN/` folder is the RelGNN repository by the RelBench team and is distributed under the [MIT License](https://github.com/snap-stanford/RelGNN/blob/main/LICENSE). Its copyright and license notice are included in `RelGNN/LICENSE`. This assignment's model code is in `BigData2026-Assignment2/scripts/native_relgnn.py` and `run_experiment.py`.
- **Data source:** The event data source is [GH Archive](https://www.gharchive.org/), which archives public GitHub events as hourly JSON files. The included Parquet tables are derived from these events and anonymized by `scripts/preprocess.py`. Consult GH Archive and GitHub's applicable terms for the source data; the RelGNN MIT software license does not license the underlying GitHub event data.

