import torch as t
import torch.nn as nn
import torch.nn.functional as F


class LowRankNet(nn.Module):
    def __init__(self, size, hidden, init_const=1.0, init_lamb=1.0,
                 init_lowrank=.0001,  act=F.relu):
        nn.Module.__init__(self)
        self.w = nn.Linear(1, hidden)
        t.nn.init.xavier_uniform(self.w.weight, init_lamb)
        # проверка на вектор или матрица
        # если сайз неизменяемый список и его длина 2
        if isinstance(size, tuple) and len(size) == 2:
            self.in_, self.out_ = size
            self.diagonal = False
        else:
            self.out_ = size
            self.diagonal = True

        self.act = act

        if self.diagonal:
            self.w_d = nn.Linear(hidden, self.out_)
            t.nn.init.xavier_uniform(self.w_d.weight, init_lowrank)
            # независимая от параметра lambda часть
            self.const = nn.Parameter(t.randn(self.out_))

        else:
            self.w_a1 = nn.Linear(hidden, self.in_)
            t.nn.init.xavier_uniform(self.w_a1.weight, init_lowrank)

            self.w_a2 = nn.Linear(hidden, self.out_)
            t.nn.init.xavier_uniform(self.w_a2.weight, init_lowrank)

            self.const = nn.Parameter(t.randn(self.in_, self.out_))
            t.nn.init.xavier_uniform(self.const,  init_const)

    def forward(self, lam):
        h = self.act(self.w(t.ones(1).to(self.w.weight.device) * lam))
        if self.diagonal:
            return self.const + self.w_d(h)
        else:
            a1 = self.w_a1(h)
            a2 = self.w_a2(h)

            return self.const + t.matmul(a1.view(-1, 1), a2.view(1, -1))


class VarLayerLowRank(nn.Module):  # вариационная однослойная сеть
    def __init__(self, in_,  out_,  hidden_num, prior_sigma=1.0, init_log_sigma=-3.0,  act=F.relu):
        nn.Module.__init__(self)
        self.mean = LowRankNet((in_, out_), hidden_num)  # параметры средних
        self.log_sigma = LowRankNet(
            (in_, out_), hidden_num)  # логарифм дисперсии
        # то же самое для свободного коэффициента
        self.mean_b = LowRankNet(out_, hidden_num)
        self.log_sigma_b = LowRankNet(out_, hidden_num)

        self.log_sigma.const.data *= 0  # забьем константу нужными нам значениями
        self.log_sigma.const.data += init_log_sigma

        self.log_sigma_b.const.data *= 0  # забьем константу нужными нам значениями
        self.log_sigma_b.const.data += init_log_sigma

        self.in_ = in_
        self.out_ = out_
        self.act = act
        self.prior_sigma = prior_sigma

    def forward(self, x, l):
        if self.training:  # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(
                self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(
                self.mean_b(l), t.exp(self.log_sigma_b(l)))

            w = self.eps_w.rsample()
            b = self.eps_b.rsample()

        else:  # во время контроля - смотрим средние значения параметра
            w = self.mean(l)
            b = self.mean_b(l)

        # функция активации
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l):
        # подсчет дивергенции
        size = self.in_, self.out_
        out = self.out_
        device = self.mean.w.weight.device
        self.eps_w = t.distributions.Normal(
            self.mean(l), t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(
            self.mean_b(l),  t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(
            t.zeros(size, device=device), t.ones(size, device=device)*self.prior_sigma)
        self.h_b = t.distributions.Normal(
            t.zeros(out, device=device), t.ones(out, device=device)*self.prior_sigma)
        k1 = t.distributions.kl_divergence(self.eps_w, self.h_w).sum()
        k2 = t.distributions.kl_divergence(self.eps_b, self.h_b).sum()
        return k1+k2
