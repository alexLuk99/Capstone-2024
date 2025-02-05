from src.main import Capstone
from config_run import runners, train_model

if __name__ == '__main__':
    capstone = Capstone(runners=runners, train_cluster_model=train_model)
    capstone.run()
