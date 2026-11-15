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

### Source code responsibilities

- `scripts/preprocess.py` reads raw hourly GH Archive JSON files, filters the cohort, constructs and anonymizes the relational tables, and writes Parquet data.
- `scripts/native_relgnn.py` loads the tables, builds time-cutoff-safe graph snapshots and features, defines event labels, and constructs the RelGNN model.
- `scripts/run_experiment.py` trains the baseline and Q2/Q3 feature variants, selects variants from validation results, trains the Q4 combined model, and produces predictions.
- `scripts/evaluate.py` checks prediction CSV format and user IDs, and can score predictions when labels are available.
- `scripts/Q1_baseline.ipynb`, `Q2_relationships.ipynb`, `Q3_temporal.ipynb`, and `Q4_combined.ipynb` provide the corresponding question-by-question notebooks. The command-line runner above is the concise reproducible entry point.

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

The output filename contains the submitted student ID (`315554001`); change the filename in `scripts/run_experiment.py` if a different ID is required. The model needs a compatible GPU for practical runtime; it falls back to CPU when CUDA is unavailable, which may take substantially longer.

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

The submitted `data/db/` folder contains the prepared tables, so this step is only needed if rebuilding them from raw GH Archive files. Put the hourly `.json.gz` archives (named `YYYY-MM-DD-H.json.gz`) in a folder, then from `BigData2026-Assignment2` run:

```powershell
python scripts/preprocess.py --raw PATH\TO\GH_ARCHIVE_HOURS --out data
```

This writes the eight Parquet tables into `data/db/`. Preprocessing requires Python 3.9+ with `pandas` and `pyarrow`. Raw GH Archive files are not included in this submission; see [GH Archive](https://www.gharchive.org/) for the public hourly event archives and access information.

## Check the prediction file

From `BigData2026-Assignment2`, check the CSV format and confirm that its user IDs match the provided test list:

```powershell
python scripts/evaluate.py --pred submission/output_315554001.csv --input submission/test_input.csv
```

The prediction CSV must have columns `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`; each activity prediction must be `0` or `1`.

## Source code, data, and licenses

- **RelGNN source code:** The `RelGNN/` folder is the RelGNN repository by the RelBench team and is distributed under the [MIT License](https://github.com/snap-stanford/RelGNN/blob/main/LICENSE). Its copyright and license notice are included in `RelGNN/LICENSE`. This assignment's model code is in `BigData2026-Assignment2/scripts/native_relgnn.py` and `run_experiment.py`.
- **Data source:** The event data source is [GH Archive](https://www.gharchive.org/), which archives public GitHub events as hourly JSON files. The included Parquet tables are derived from these events and anonymized by `scripts/preprocess.py`. Consult GH Archive and GitHub's applicable terms for the source data; the RelGNN MIT software license does not license the underlying GitHub event data.

