from torch.utils.data import DataLoader
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config

    def test_dataloader(self) -> DataLoader:
        trials = np.loadtxt(self.config['trial_path'], str)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))

        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        # load any dataset this one "Evaluation_Dataset" is dummy 
        eval_dataset = Evaluation_Dataset(eval_path, root="./wav/")
        loader = DataLoader(eval_dataset, num_workers=10, shuffle=False, batch_size=1)

        return loader