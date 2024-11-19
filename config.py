from torch import nn


class Config:
    config = {
        "train_params": {
            "distance_dim": 1,
            "lam1": 0.1,
            "lam2": 0.1,
            "lam3": 0.1,
            "stage1_iter": 20,
            "covariate_iter": 20,
            "mi_iter": 20,
            "odds_iter": 100,
            "stage2_iter": 1,
            "lam4": 0.1,
            "n_epoch": 100,
            "treatment_weight_decay": 0.0,
            "instrumental_weight_decay": 0.0,
            "covariate_weight_decay": 0.1,
            "s1_weight_decay": 0.0,
            "odds_weight_decay": 0.0,
            "selection_weight_decay": 0.0,
            "r0_weight_decay": 0.0,
            "r1_weight_decay": 0.0
        }
    }
    experiment_num = 50
    c_strength = 10
    u_strength = 10
    sample_num = 5000

    def __init__(self):
        self.networks = self.initialize_model_structure()

    def initialize_model_structure(self):
        networks = [
            nn.Sequential(nn.Linear(1, 1)),

            nn.Sequential(nn.Linear(3, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 3),
                          nn.BatchNorm1d(3)),

            nn.Sequential(nn.Linear(5, 128),
                          nn.BatchNorm1d(128),
                          nn.ReLU(),
                          nn.Linear(128, 1),
                          nn.Sigmoid()),

            nn.Sequential(nn.Linear(2, 128),
                          nn.ReLU(),
                          nn.Linear(128, 32),
                          nn.BatchNorm1d(32),
                          nn.ReLU(),
                          nn.Linear(32, 2),
                          nn.ReLU()),

            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          ),

            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          ),

            nn.Sequential(nn.Linear(3, 1),
                          nn.Sigmoid()
                          ),

            nn.Sequential(nn.Linear(4, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Sigmoid()),

            nn.Sequential(nn.Linear(5, 16),
                          nn.BatchNorm1d(16),
                          nn.ReLU(),
                          nn.Linear(16, 1),
                          nn.Softplus()
                          ),
        ]
        return networks
