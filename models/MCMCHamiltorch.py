from hamiltorch.samplers import define_model_log_prob
from models.base_model import RegressionMLP, EvaluationModel
from utilities.data_generation import RegressionData
import torch
import hamiltorch


class MCMCReg(EvaluationModel):
    def __init__(self,
                 input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, loss_fct: callable, n_hidden: int = 4,
                 n_samples: int = 1000, test_n_samples: int = 1000,
                 wandb_active: bool = False, num_samples: int = 1000,
                 heteroscedastic: bool = False, train_interval: tuple = (-2, 2),
                 test_interval: tuple = (-3, 3), problem: str = 'regression',
                 add_sigmoid: bool = False, seed: int = 42, step_size: float = 0.0005,
                 num_steps_per_sample: int = 30, burn: int = -1, tau: float = 1.0,
                 **kwargs):

        self.base_model = RegressionMLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                        output_dim=output_dim, n_hidden=n_hidden,
                                        add_sigmoid=add_sigmoid, seed=seed)

        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None
        self.test_mse = None
        self.variance_predictions = None
        self.params_hmc_gpu = None
        self.tau_list = None
        self.tau_out = None
        self.lower = None
        self.upper = None
        self.lower_al = None
        self.upper_al = None
        self.mean_predictions = None
        self.standard_deviation_predictions = None
        self.standard_deviation_aleatoric = None
        self.test_data_set = None
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       test_interval=test_interval, train_interval=train_interval,
                                       problem=problem, heteroscedastic=heteroscedastic)
        self.reg_fct = reg_fct

        self.model_loss = None
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.num_samples = num_samples
        self.wandb_active = wandb_active
        self.problem = problem
        self.loss_function = loss_fct
        self.training_time = None
        self.step_size = step_size
        self.num_steps_per_sample = num_steps_per_sample
        self.burn = burn
        self.tau = tau

    def train_model(self, store_on_GPU: bool = False, debug: bool = False, model_loss: str = 'regression',
                    mass: float = 1.0,  tau_out: float = 2.0, r: int = 0, **kwargs):
        """ set tau to 1000 for a less bendy function """
        device = 'cpu'
        tau_list = []
        for w in self.base_model.parameters():
            tau_list.append(self.tau)  # set the prior precision to be the same for each set of weights
        # this sets the prior distribution for the weights
        tau_list = torch.tensor(tau_list).to(device)
        # Set initial weights
        params_init = hamiltorch.util.flatten(self.base_model).to(device).clone()
        # Set the Inverse of the Mass matrix
        inv_mass = torch.ones(params_init.shape) / mass

        integrator = hamiltorch.Integrator.EXPLICIT
        sampler = hamiltorch.Sampler.HMC

        hamiltorch.set_random_seed(r)
        params_hmc_f = hamiltorch.sample_model(self.base_model,
                                               x=torch.Tensor(self.data_set.train_data.x).view(-1, 1).to(device),
                                               y=torch.Tensor(self.data_set.train_data.y).view(-1, 1).to(device),
                                               params_init=params_init,
                                               model_loss=self.loss_function, num_samples=self.num_samples,
                                               burn=self.burn, inv_mass=inv_mass.to(device), step_size=self.step_size,
                                               num_steps_per_sample=self.num_steps_per_sample, tau_out=tau_out,
                                               tau_list=tau_list,
                                               debug=debug, store_on_GPU=store_on_GPU,
                                               sampler=sampler)

        params_hmc_gpu = [ll.to(device) for ll in params_hmc_f[1:]]

        # Let's evaluate the performance over the training data
        pred_list_tr, pred_var, log_probs_split_tr = self.make_predictions(self.base_model,
                                                                           x=torch.Tensor(
                                                                               self.data_set.train_data.x).view(-1,
                                                                                                                1).to(
                                                                               device),
                                                                           y=torch.Tensor(
                                                                               self.data_set.train_data.y).view(-1,
                                                                                                                1).to(
                                                                               device),
                                                                           samples=params_hmc_gpu,
                                                                           model_loss=self.loss_function,
                                                                           tau_out=tau_out, tau_list=tau_list)
        ll_full = torch.zeros(pred_list_tr.shape[0])
        # ll_full[0] = - 0.5 * tau_out * (
        #             (pred_list_tr[0].cpu() - torch.Tensor(self.data_set.train_data.y).view(-1, 1).to(device)) ** 2).sum(
        #     0)
        # for i in range(pred_list_tr.shape[0]):
        #     ll_full[i] = - 0.5 * tau_out * ((pred_list_tr[:i].mean(0).cpu() - torch.Tensor(
        #         self.data_set.train_data.y).view(-1, 1).to(device)) ** 2).sum(0)
        #
        # self.model_loss = ll_full
        self.params_hmc_gpu = params_hmc_gpu
        self.tau_list = tau_list
        self.tau_out = tau_out

    def make_predictions_on_test(self, classification_type: str = 'two_outputs'):
        device = 'cpu'
        # Let's predict over the entire test range [-2,2]
        pred_list, pred_var, log_probs_f = self.make_predictions(self.base_model,
                                                                 x=torch.Tensor(self.data_set.test_data.x).view(-1,
                                                                                                                1).to(
                                                                     device),
                                                                 y=torch.Tensor(self.data_set.test_data.y).view(-1,
                                                                                                                1).to(
                                                                     device),
                                                                 samples=self.params_hmc_gpu,
                                                                 model_loss=self.loss_function,
                                                                 tau_out=self.tau_out,
                                                                 tau_list=self.tau_list)

        if self.problem == 'regression':
            self.mean_predictions = pred_list[200:].mean(0).to('cpu').detach().numpy()
            self.aleatoric_uncertainty = pred_var[200:].mean(0).to('cpu').detach().numpy()
            self.epistemic_uncertainty = pred_list[200:].var(0).to('cpu').detach().numpy()
        else:
            if classification_type == 'two_outputs':
                # put mean and variance in the same tensor
                predictions = torch.cat([pred_list, pred_var], dim=2)
                self.accuracy = self.get_accuracy(self.data_set.test_data.y, predictions, stacked=True)
            else:
                self.get_information_theoretical_decomposition(pred_list, stacked=True)

    def make_predictions_on_test_classification(self):
        self.make_predictions_on_test()

    def make_predictions_on_test_classification_information(self):
        self.make_predictions_on_test(classification_type='one_output')

    def make_predictions(self, model, samples, x, y, model_loss: callable, tau_out=1., tau_list=None, verbose=False):
        with torch.no_grad():
            params_shape_list = []
            params_flattened_list = []
            build_tau = False
            if tau_list is None:
                tau_list = []
                build_tau = True
            for weights in model.parameters():
                params_shape_list.append(weights.shape)
                params_flattened_list.append(weights.nelement())
                if build_tau:
                    tau_list.append(torch.tensor(1.))

            if x is not None and y is not None:

                if x.device != samples[0].device:
                    raise RuntimeError('x on device: {} and samples on device: {}'.format(x.device, samples[0].device))

                log_prob_func = define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list,
                                                      tau_list, tau_out, predict=True, device=samples[0].device)

                pred_log_prob_list = []
                pred_list = []
                pred_list_var = []
                for s in samples:
                    lp, pred = log_prob_func(s)
                    pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
                    num_outputs = pred.shape[1]
                    if num_outputs >= 2:
                        if self.problem == 'regression':
                            mean = pred[:, 0]
                            var = torch.exp(pred[:, 1])
                        else:
                            mean = pred[:, :int(pred.shape[1] / 2)]
                            var = torch.exp(pred[:, int(pred.shape[1] / 2):])
                        pred_list.append(mean.detach())
                        pred_list_var.append(var.detach())
                    else:
                        pred_list.append(pred.detach())
            else:
                raise RuntimeError('Val data not defined (i.e. arguments x, y, val_loader are all not defined)')
        # var stack is empty for one output models
        var_stack = torch.stack(pred_list_var) if pred_list_var else pred_list_var
        return torch.stack(pred_list), var_stack, pred_log_prob_list
