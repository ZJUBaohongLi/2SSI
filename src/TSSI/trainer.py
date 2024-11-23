from __future__ import annotations
from itertools import chain
from typing import Dict, List, Any, Optional
from torch.utils.tensorboard import SummaryWriter

import numpy

from src.utils.mi_estimators import *

from src.TSSI.model import TSSIModel
from src.data.data_generation import demand
from src.data.data_class import TrainDataSetTorch, TestDataSetTorch, concat_dataset
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred
from config import Config


class TSSITrainer(object):

    def __init__(self, networks: List[Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False):
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        # configure training params
        self.lam1: float = train_params["lam1"]
        self.lam2: float = train_params["lam2"]
        self.lam3: float = train_params["lam3"]
        self.lam4: float = train_params["lam4"]
        self.distance_dim: int = train_params["distance_dim"]
        self.stage1_iter: int = train_params["stage1_iter"]
        self.stage2_iter: int = train_params["stage2_iter"]
        self.covariate_iter: int = train_params["covariate_iter"]
        self.mi_iter: int = train_params["mi_iter"]
        self.odds_iter: int = train_params["odds_iter"]
        self.n_epoch: int = train_params["n_epoch"]
        self.add_stage1_intercept: bool = True
        self.add_stage2_intercept: bool = True
        self.treatment_weight_decay: float = train_params["treatment_weight_decay"]
        self.instrumental_weight_decay: float = train_params["instrumental_weight_decay"]
        self.covariate_weight_decay: float = train_params["covariate_weight_decay"]
        self.selection_weight_decay: float = train_params["selection_weight_decay"]
        self.r1_weight_decay: float = train_params["r1_weight_decay"]
        self.r0_weight_decay: float = train_params["r0_weight_decay"]
        self.s1_weight_decay: float = train_params["s1_weight_decay"]
        self.odds_weight_decay: float = train_params["odds_weight_decay"]

        # build networks
        self.treatment_net: nn.Module = networks[0]
        self.instrumental_net: nn.Module = networks[1]
        self.selection_net: nn.Module = networks[2]
        self.covariate_net: Optional[nn.Module] = networks[3]
        self.r1_net: nn.Module = networks[4]
        self.r0_net: nn.Module = networks[5]
        self.phi_net: nn.Module = networks[6]
        self.s1_net: nn.Module = networks[7]
        self.odds_net: nn.Module = networks[8]

        if self.gpu_flg:
            self.treatment_net.to("cuda:0")
            self.instrumental_net.to("cuda:0")
            if self.covariate_net is not None:
                self.covariate_net.to("cuda:0")
            self.selection_net.to("cuda:0")
            self.r1_net.to("cuda:0")
            self.r0_net.to("cuda:0")
            self.phi_net.to("cuda:0")
            self.s1_net.to("cuda:0")
            self.odds_net.to("cuda:0")

        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),
                                              weight_decay=self.treatment_weight_decay)
        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),
                                                 weight_decay=self.instrumental_weight_decay)
        self.s1_opt = torch.optim.Adam(chain(self.s1_net.parameters(), self.phi_net.parameters()),
                                       weight_decay=self.s1_weight_decay)

        if self.covariate_net:
            self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),
                                                  weight_decay=self.covariate_weight_decay)

    def train(self, rand_seed: int = 42, verbose: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data, unselected_train_data, test_data, unselected_test_data = demand(Config.sample_num * 5,
                                                                                        rand_seed)
        train_1st_t, train_2nd_t, train_3rd_t = concat_dataset(train_data,
                                                               unselected_train_data), train_data, unselected_train_data
        train_1st_t = TrainDataSetTorch.from_numpy(train_1st_t)
        train_2nd_t = TrainDataSetTorch.from_numpy(train_2nd_t)
        train_3rd_t = TrainDataSetTorch.from_numpy(train_3rd_t)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        unselected_test_data_t = TestDataSetTorch.from_numpy(unselected_test_data)
        if self.gpu_flg:
            train_1st_t = train_1st_t.to_gpu()
            train_2nd_t = train_2nd_t.to_gpu()
            train_3rd_t = train_3rd_t.to_gpu()
            test_data_t = test_data_t.to_gpu()
            unselected_test_data_t = unselected_test_data_t.to_gpu()

        self.lam1 *= train_1st_t[0].size()[0]
        self.lam2 *= train_2nd_t[0].size()[0]
        self.lam3 *= train_3rd_t[0].size()[0]
        writer = SummaryWriter()
        for t in range(self.n_epoch):
            self.stage1_update(train_1st_t, t, writer)
            if self.covariate_net:
                self.update_covariate_net(train_1st_t, train_2nd_t, t, writer)
            self.stage2_update(train_1st_t, train_2nd_t, t, writer)
        writer.close()
        mdl = TSSIModel(self.treatment_net, self.instrumental_net, self.selection_net,
                        self.covariate_net, self.r1_net, self.r0_net, self.odds_net, self.phi_net,
                        self.add_stage1_intercept, self.add_stage2_intercept,
                        self.odds_iter, self.selection_weight_decay,
                        self.r1_weight_decay, self.r0_weight_decay, self.odds_weight_decay)
        mdl.fit_t(train_1st_t, train_2nd_t, train_3rd_t, self.lam1, self.lam2, self.lam3)

        if self.gpu_flg:
            torch.cuda.empty_cache()

        oos_loss: numpy.ndarray = mdl.evaluate_t(test_data_t)
        unselected_loss: numpy.ndarray = mdl.evaluate_t(unselected_test_data_t)
        return oos_loss, unselected_loss

    def stage1_update(self, train_1st_t: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        self.treatment_net.train(False)
        self.instrumental_net.train(True)
        self.phi_net.train(True)
        self.s1_net.train(True)
        bce_func = nn.BCELoss()
        if self.covariate_net:
            self.covariate_net.train(False)
        mi_estimator = eval("CLUB")(self.distance_dim, self.distance_dim, self.distance_dim * 2)
        if self.gpu_flg:
            mi_estimator = mi_estimator.to("cuda:0")
        mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-4)
        treatment_feature = self.treatment_net(train_1st_t.treatment).detach()
        for i in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            instrumental_feature = self.instrumental_net(train_1st_t.instrumental)
            feature_t = TSSIModel.augment_stage1_feature(instrumental_feature,
                                                         self.add_stage1_intercept)
            loss_t = linear_reg_loss(treatment_feature, feature_t, self.lam1)
            loss_t.backward()
            self.instrumental_opt.step()
            writer.add_scalar('InstrumentalNet Train loss', loss_t, epoch * self.stage1_iter + i)
            for j in range(self.mi_iter):
                mi_estimator.train(True)
                phi_feature = self.phi_net(train_1st_t.instrumental)
                mi_loss = mi_estimator.learning_loss(phi_feature, train_1st_t.treatment)
                mi_optimizer.zero_grad()
                mi_loss.backward()
                mi_optimizer.step()
            mi_estimator.train(False)
            phi_feature = self.phi_net(train_1st_t.instrumental)
            s_pred = self.s1_net(torch.cat((train_1st_t.treatment, train_1st_t.covariate, phi_feature), 1))
            loss_s = bce_func(s_pred, train_1st_t.selection) + self.lam4 * mi_estimator(phi_feature, train_1st_t.treatment)
            self.s1_opt.zero_grad()
            loss_s.backward()
            self.s1_opt.step()
            writer.add_scalar('Phi Train loss', loss_s, epoch * self.stage1_iter + i)

    def stage2_update(self, train_1st_t: TrainDataSetTorch, train_2nd_t: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        self.treatment_net.train(True)
        self.instrumental_net.train(False)
        self.phi_net.train(False)
        if self.covariate_net:
            self.covariate_net.train(False)

        # have instrumental features
        instrumental_1st_feature = self.instrumental_net(train_1st_t.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_t.instrumental).detach()

        phi_2nd_feature = self.phi_net(train_2nd_t.instrumental).detach()

        covariate_2nd_feature = None
        # have covariate features
        if self.covariate_net:
            covariate_2nd_feature = self.covariate_net(train_2nd_t.covariate).detach()

        for i in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            treatment_1st_feature = self.treatment_net(train_1st_t.treatment)
            treatment_2nd_feature = self.treatment_net(train_2nd_t.treatment)
            res = TSSIModel.fit_2sls(treatment_1st_feature,
                                     treatment_2nd_feature,
                                     instrumental_1st_feature,
                                     instrumental_2nd_feature,
                                     phi_2nd_feature,
                                     covariate_2nd_feature,
                                     train_2nd_t.outcome,
                                     self.lam1, self.lam2,
                                     self.add_stage1_intercept,
                                     self.add_stage2_intercept)
            loss = res["stage2_loss"]
            loss.backward()
            self.treatment_opt.step()
            writer.add_scalar('TreatmentNet Train loss', loss, epoch * self.stage1_iter + i)

    def update_covariate_net(self, train_1st_data: TrainDataSetTorch, train_2nd_data: TrainDataSetTorch, epoch: int, writer: SummaryWriter):
        # have instrumental features
        self.instrumental_net.train(False)
        instrumental_1st_feature = self.instrumental_net(train_1st_data.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_data.instrumental).detach()

        self.treatment_net.train(False)
        treatment_1st_feature = self.treatment_net(train_1st_data.treatment).detach()
        treatment_2nd_feature = self.treatment_net(train_2nd_data.treatment).detach()

        feature = TSSIModel.augment_stage1_feature(instrumental_1st_feature, self.add_stage1_intercept)
        stage1_weight = fit_linear(treatment_1st_feature, feature, self.lam1)

        # residual for stage 2
        feature = TSSIModel.augment_stage1_feature(instrumental_2nd_feature,
                                                   self.add_stage1_intercept)
        predicted_treatment_feature = linear_reg_pred(feature, stage1_weight).detach()
        residual_2nd_feature = treatment_2nd_feature - predicted_treatment_feature

        self.covariate_net.train(True)
        self.phi_net.train(False)
        phi_feature = self.phi_net(train_2nd_data.instrumental).detach()
        condition_feature = torch.concat((residual_2nd_feature, phi_feature), 1)
        for i in range(self.covariate_iter):
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_2nd_data.covariate)
            # stage2 - y1 regression
            feature = TSSIModel.augment_stage_y1_feature(treatment_2nd_feature,
                                                         condition_feature,
                                                         covariate_feature,
                                                         self.add_stage2_intercept)

            loss = linear_reg_loss(train_2nd_data.outcome, feature, self.lam2)
            loss.backward()
            self.covariate_opt.step()
            writer.add_scalar('CovariateNet Train loss', loss, epoch * self.stage1_iter + i)
