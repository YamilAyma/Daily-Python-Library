from scripts import generate_workflows_pointblank as pb

# Use: Put your python scripts in the currently directory and execute -- with darts_utils.py
path_csv = "./utils/data.csv"
# df = pb._load_dataframe_from_csv(path_csv)

# pb.generate_baseline_yaml(df, "output.yaml", "my_table", path_csv)
pb.run_validation_from_yaml("output.yaml")