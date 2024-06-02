from pathlib import Path

input_path = Path('data/raw')
interim_path = Path('data/interim')
output_path = Path('data/output')
models_path = Path('data/models')

input_path.mkdir(exist_ok=True, parents=True)
interim_path.mkdir(exist_ok=True, parents=True)
output_path.mkdir(exist_ok=True, parents=True)
models_path.mkdir(exist_ok=True, parents=True)
