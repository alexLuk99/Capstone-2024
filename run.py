from src.main import Capstone
from config_run import runners

capstone = Capstone(runners=runners)
capstone.run()
