from typing import NamedTuple, Optional
import numpy as np
import torch


class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray
    selection: Optional[np.ndarray]


class TestDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: Optional[np.ndarray]
    covariate: Optional[np.ndarray]
    structural: np.ndarray


class TrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor
    selection: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: TrainDataSet):
        covariate = None
        selection = None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=torch.float32)
        if train_data.selection is not None:
            selection = torch.tensor(train_data.selection, dtype=torch.float32)
        return TrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                 instrumental=torch.tensor(train_data.instrumental, dtype=torch.float32),
                                 covariate=covariate,
                                 outcome=torch.tensor(train_data.outcome, dtype=torch.float32),
                                 structural=torch.tensor(train_data.structural, dtype=torch.float32),
                                 selection=selection)

    def to_gpu(self):
        covariate = None
        selection = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        if self.selection is not None:
            selection = self.selection.cuda()
        return TrainDataSetTorch(treatment=self.treatment.cuda(),
                                 instrumental=self.instrumental.cuda(),
                                 covariate=covariate,
                                 outcome=self.outcome.cuda(),
                                 structural=self.structural.cuda(),
                                 selection=selection)


class TestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: TestDataSet):
        covariate = None
        instrumental = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=torch.float32)
        if test_data.instrumental is not None:
            instrumental = torch.tensor(test_data.instrumental, dtype=torch.float32)
        return TestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                instrumental=instrumental,
                                covariate=covariate,
                                structural=torch.tensor(test_data.structural, dtype=torch.float32))

    def to_gpu(self):
        covariate = None
        instrumental = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        if self.instrumental is not None:
            instrumental = self.instrumental.cuda()
        return TestDataSetTorch(treatment=self.treatment.cuda(),
                                instrumental=instrumental,
                                covariate=covariate,
                                structural=self.structural.cuda())


def concat_dataset(dataset1: TrainDataSet, dataset2: TrainDataSet):
    new_t = np.concatenate((dataset1.treatment, dataset2.treatment), axis=0)
    new_z = np.concatenate((dataset1.instrumental, dataset2.instrumental), axis=0)
    new_x = np.concatenate((dataset1.covariate, dataset2.covariate), axis=0)
    new_y = np.concatenate((dataset1.outcome, dataset2.outcome), axis=0)
    new_gt = np.concatenate((dataset1.structural, dataset2.structural), axis=0)
    new_s = np.concatenate((dataset1.selection, dataset2.selection), axis=0)
    return TrainDataSet(treatment=new_t,
                        instrumental=new_z,
                        covariate=new_x,
                        outcome=new_y,
                        structural=new_gt,
                        selection=new_s
                        )
