from pathlib import Path

input_path = Path('data/raw')
interim_path = Path('data/interim')
output_path = Path('data/output')
models_path = Path('data/models')

models_path.mkdir(exist_ok=True, parents=True)
