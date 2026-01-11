# Basic Decision Tree Project

A small example project that trains a Decision Tree regressor on housing data and reports prediction results and the model's mean absolute error.

## Contents
- `basiclearningdata.py` — trains the model, evaluates different tree sizes, and prints predictions.
- `Housing.csv` — expected dataset (place in the same folder or update the path in the script).

## Requirements
- Python 3.8+ (tested)
- pandas
- scikit-learn

Install dependencies with:

```powershell
pip install pandas scikit-learn
```

## Usage
1. Ensure `Housing.csv` is available in the project root (or update the `housing_file_path` inside `basiclearningdata.py`).
2. Run the project using a python powershell command or through a source code editor

The script will:
- Load the dataset
- Split into train/validation sets
- Search for the best `max_leaf_nodes` value (by MAE)
- Train a final Decision Tree model and print a few predictions and the margin of error

## Notes
- If the dataset path is incorrect, update the `housing_file_path` variable at the top of `basiclearningdata.py` (currently set to `/Housing.csv`).
- To save environment dependencies, run `pip freeze > requirements.txt` and include it in the repo.
