from src.main import Capstone
from config_run import runners

if __name__ == '__main__':
    capstone = Capstone(runners=runners)
    capstone.run()
