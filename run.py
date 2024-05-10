from config_run import runners
from src.main import Capstone

capstone = Capstone(runners=runners)
capstone.load_data()
