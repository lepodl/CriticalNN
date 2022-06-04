import os
import prettytable as pt
from dtb.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from brain_block.bold_model_pytorch import BOLD
import time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as mp
import argparse

mp.switch_backend('Agg')


class DA_MODEL:
    def __init__(self, block_dict: dict, bold_dict: dict, steps=800, ensembles=100, time=400, hp_sigma=1.,
                 bold_sigma=1e-6):
        """
        Mainly for the whole brain model consisting of cortical functional column structure and canonical E/I=4:1 structure.

       Parameters
       ----------
       block_name: str
           block name.
       block_dict : dict
           contains the parameters of the block model.
       bold_dict : dict
           contains the parameters of the bold model.
        """
        self.block = block_gpu(block_dict['ip'], block_dict['block_path'], block_dict['delta_t'],
                               route_path=None, force_rebase=True, cortical_size=1)  # !!!!
        self.noise_rate = block_dict['noise_rate']
        self.delta_t = block_dict['delta_t']
        self.bold = BOLD(bold_dict['epsilon'], bold_dict['tao_s'], bold_dict['tao_f'], bold_dict['tao_0'],
                                  bold_dict['alpha'], bold_dict['E_0'], bold_dict['V_0'])
        self.ensembles = ensembles
        self.num_populations = int(self.block.total_subblks)
        print("num_populations", self.num_populations)
        self.num_populations_in_one_ensemble = int(self.num_populations / self.ensembles)
        self.num_neurons = int(self.block.total_neurons)
        self.num_neurons_in_one_ensemble = int(self.num_neurons / self.ensembles)
        self.populations_id = self.block.subblk_id.cpu().numpy()
        print("len(populations_id)", len(self.populations_id))
        self.neurons = self.block.neurons_per_subblk
        self.populations_id_per_ensemble = np.split(self.populations_id, self.ensembles)

        self.T = time
        self.steps = steps
        self.hp_sigma = hp_sigma
        self.bold_sigma = bold_sigma

    @staticmethod
    def log(val, low_bound, up_bound, scale=10):
        assert torch.all(torch.le(val, up_bound)) and torch.all(
            torch.ge(val, low_bound)), "In function log, input data error!"
        return scale * (torch.log(val - low_bound) - torch.log(up_bound - val))

    @staticmethod
    def sigmoid(val, low_bound, up_bound, scale=10):
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(val / scale)
        elif isinstance(val, np.ndarray):
            assert np.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(
                torch.from_numpy(val.astype(np.float32)) / scale).numpy()
        else:
            raise ValueError("val type is wrong!")

    @staticmethod
    def torch_2_numpy(u, is_cuda=True):
        assert isinstance(u, torch.Tensor)
        if is_cuda:
            return u.cpu().numpy()
        else:
            return u.numpy()

    @staticmethod
    def show_bold(W, bold, T, path, brain_num):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(5, 3), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold[:T, i], 'r-')
            ax1.plot(iteration, np.mean(W[:T, :, i, -1], axis=1), 'b-')
            mp.fill_between(iteration, np.mean(W[:T, :, i, -1], axis=1) -
                            np.std(W[:T, :, i, -1], axis=1), np.mean(W[:T, :, i, -1], axis=1)
                            + np.std(W[:T, :, i, -1], axis=1), color='b', alpha=0.2)
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)
        return None

    @staticmethod
    def show_hp(hp, T, path, brain_num, hp_num, hp_real=None):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(5, 3), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is None:
                    pass
                else:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path, "hp" + str(i) + "_" + str(j) + ".png"), bbox_inches='tight', pad_inches=0)
                mp.close(fig)
        return None

    def initial_model(self, real_parameter, para_ind):
        """
        initialize the block model, and then determine the random walk range of hyper parameter,
        -------
        """
        raise NotImplementedError

    def evolve(self, steps=800):
        """
        evolve the block model and obtain prediction observation,
        here we apply the MC method to evolve samples drawn from initial Gaussian distribution.
        -------

        """
        raise NotImplementedError

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        use kalman filter to filter the latent state.
        -------

        """
        raise NotImplementedError


class DA_Rest_Whole_Brain(DA_MODEL):
    def __init__(self, block_dict: dict, bold_dict: dict, whole_brain_voxel_info: str, steps, ensembles, time, hp_sigma,
                 bold_sigma):
        super().__init__(block_dict, bold_dict, steps, ensembles, time, hp_sigma, bold_sigma)
        self.device = "cuda:0"
        # whole_brain_voxel_info = '/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/jianfeng_normal/A1_1_DTI_voxel_structure_data_jianfeng.mat'
        # file = h5py.File(whole_brain_voxel_info, 'r')
        # aal_region = file['dti_AAL'][:]
        # aal_region = aal_region[0].astype(np.int32)
        aal_region = np.arange(90, dtype=np.int8)
        self.num_voxel_in_one_ensemble = len(aal_region)
        assert self.num_populations_in_one_ensemble == self.num_voxel_in_one_ensemble * 2
        assert self.populations_id.max() == self.num_populations_in_one_ensemble * self.ensembles - 1, "population_id is not correct!"
        self.num_voxel = int(self.num_populations / 2)
        self.neurons_per_voxel = self.block.neurons_per_subblk.cpu().numpy()

        neurons_per_voxel, _ = np.histogram(self.populations_id, weights=self.neurons_per_voxel, bins=self.num_voxel,
                                            range=(0, self.num_voxel * 2))
        self.neurons_per_voxel = neurons_per_voxel

        self.hp_num = None
        self.up_bound = None
        self.low_bound= None
        self.hp = None
        self.hp_log= None

    def __str__(self):
        print("DA FOR REST WHOLE BRAIN")

    @staticmethod
    def random_walk_range(real_parameter, voxels, up_times=3., low_times=2.):
        if real_parameter.ndim == 2:
            temp_up = np.tile(real_parameter * up_times, (voxels, 1))
            temp_low = np.tile(real_parameter / low_times, (voxels, 1))
            return temp_up.reshape((voxels, -1)), temp_low.reshape((voxels, -1))
        else:
            raise NotImplementedError("real_parameter.ndim=1 is waiting to complete")

    def update(self, e_rate, i_rate):
        para_base = torch.tensor([0.02675, 0.004, 0.5034, 0.02775], dtype=torch.float32)
        population_info = np.stack(np.meshgrid(self.populations_id, np.array([10]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[0] * e_rate
        ampa = torch.repeat_interleave(ampa, 2)
        self.block.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(self.populations_id, np.array([11]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[1] * (1 - e_rate)
        ampa = torch.repeat_interleave(ampa, 2)
        self.block.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(self.populations_id, np.array([12]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[2] * i_rate
        ampa = torch.repeat_interleave(ampa, 2)
        self.block.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(self.populations_id, np.array([13]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[3] * (1 - i_rate)
        ampa = torch.repeat_interleave(ampa, 2)
        self.block.mul_property_by_subblk(population_info, ampa.reshape(-1))

    def initial_model(self, real_parameter, para_ind, up_times=3, low_times=2):
        start = time.time()
        self.hp_num = 2
        for idx in np.array([10, 11, 12, 13]):
            population_info = np.stack(np.meshgrid(self.populations_id, idx, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            gamma = torch.ones(self.num_populations, device="cuda:0") * 5.
            self.block.gamma_property_by_subblk(population_info, gamma, gamma, debug=False)

        # CPU numpy to GPU ndarray
        real_parameter = np.array([0.4, 0.25]).reshape((2, 1))
        self.up_bound, self.low_bound = self.random_walk_range(real_parameter,self.num_voxel_in_one_ensemble, up_times=2.4, low_times=2)

        self.hp = np.random.uniform(self.low_bound, self.up_bound, size=(self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num)).astype(np.float32)

        print(f"in initial_model, hp shape {self.hp.shape}")
        self.hp = torch.from_numpy(self.hp).cuda()

        self.up_bound, self.low_bound = torch.from_numpy(self.up_bound.astype(np.float32)).cuda(), torch.from_numpy(self.low_bound.astype(np.float32)).cuda()
        self.update(self.hp[:, :, 0].reshape(-1), self.hp[:, :, 1].reshape(-1))
        print(f"=================Initial DA MODEL done! cost time {time.time() - start:.2f}==========================")

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        distributed ensemble kalman filter. for single voxel, it modified both by its own observation and also
        by other's observation with distributed rate.

        Parameters
        ----------
        w_hat  :  store the state matrix, shape=(ensembles, voxels, states)
        bold_y_t : shape=(voxels)
        rate : distributed rate
        """
        ensembles, voxels, total_state_num = w_hat.shape  # ensemble, brain_n, hp_num+act+hemodynamic(total state num)
        assert total_state_num == self.hp_num + 6
        w = w_hat.clone()
        w_mean = torch.mean(w_hat, dim=0, keepdim=True)
        w_diff = w_hat - w_mean
        w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
        w_cxx = torch.sum(w_cx, dim=0) / (self.ensembles - 1) + self.bold_sigma
        temp = w_diff[:, :, -1] / (w_cxx.reshape([1, voxels])) / (self.ensembles - 1)  # (ensemble, brain)
        model_noise = self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, voxels)).type_as(temp)
        w += rate * (bold_y_t + model_noise - w_hat[:, :, -1])[:, :, None] * torch.sum(
            temp[:, :, None] * w_diff.reshape([self.ensembles, voxels, total_state_num]), dim=0, keepdim=True)
        # print("min1:", torch.min(w[:, :, :self.hp_num]).item(), "max1:", torch.max(w[:, :, :self.hp_num]).item())
        w += (1 - rate) * torch.mm(torch.mm(bold_y_t + model_noise - w_hat[:, :, -1], temp.T) / voxels,
                                   w_diff.reshape([self.ensembles, voxels * total_state_num])).reshape(
            [self.ensembles, voxels, total_state_num])
        print("after filter", "hp_log_min:",
              torch.min(w[:, :, :self.hp_num]).item(), "hp_log_max:",
              torch.max(w[:, :, :self.hp_num]).item())
        return w

    def evolve(self, steps=800):
        print(f"evolve:")
        start = time.time()
        out = None
        steps = steps if steps is not None else self.steps
        act_all = []
        for act in self.block.run(steps*10, freqs=True, vmean=False, sample_for_show=False):
            act = act.cpu().numpy()
            act_all.append(act)
        act_all = np.array(act_all).reshape((steps, 10, -1))
        act_all = np.sum(act_all, axis=1)
        for idxx in range(steps):
            act = act_all[idxx]
            active,  _ = np.histogram(self.populations_id, weights=act, bins=self.num_voxel,
                                            range=(0, self.num_voxel * 2))
            active = (active / self.neurons_per_voxel).reshape(-1)
            active = torch.from_numpy(active.astype(np.float32)).cuda()
            out = self.bold.run(torch.max(active, torch.tensor([1e-05]).type_as(active)))

        print(
            f'active: {active.mean().item():.3f},  {active.min().item():.3f} ------> {active.max().item():.3f}')

        bold = torch.stack(
            [self.bold.s, self.bold.q, self.bold.v, self.bold.f_in, out])

        # print("cortical max bold_State: s, q, v, f_in, bold ", bold1.max(1)[0].data)
        print("bold range:", bold[-1].min().data, "------>>", bold[-1].max().data)

        w = torch.cat((self.hp_log, active.reshape([self.ensembles, -1, 1]),
                        bold.T.reshape([self.ensembles, -1, 5])), dim=2)
        print(f'End evolving, totally cost time: {time.time() - start:.2f}')
        return w

    def run(self, real_parameter, para_ind, bold_path="whole_brain_voxel_info.npz", write_path="./"):
        total_start = time.time()

        tb = pt.PrettyTable()
        tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
        tb.add_row([1, "name", "da_rest_brain_with_new_init", "ensembles", self.ensembles])
        tb.add_row([2, "total_populations", self.num_populations, "populations_per_ensemble",
                    self.num_populations_in_one_ensemble])
        tb.add_row([3, "total_neurons", self.num_neurons, "neurons_per_ensemble", self.num_neurons_in_one_ensemble])
        tb.add_row([4, "voxels_per_ensemble", self.num_voxel_in_one_ensemble, "populations_per_voxel", "2"])
        tb.add_row([5, "total_time", self.T, "steps", self.steps])
        tb.add_row([6, "hp_sigma", self.hp_sigma, "bold_sigma", self.bold_sigma])
        tb.add_row([7, "noise_rate", self.noise_rate, "bold_range", "0.02-0.05"])
        tb.add_row([8, "walk_upbound", "x3", "walk_low_bound", "/2"])
        print(tb)

        self.initial_model(real_parameter, para_ind, up_times=3, low_times=2)
        self.hp_log = self.log(self.hp, self.low_bound, self.up_bound)
        print("self.hp.dtype", self.hp.dtype)
        print("self.hp_log.dtype", self.hp_log.dtype)

        _ = self.evolve(steps=800)
        _ = self.evolve(steps=800)
        _ = self.evolve(steps=800)
        w = self.evolve(steps=800)

        bold_y = h5py.File(bold_path, 'r')['dti_rest_state'][:]
        bold_y = bold_y[:self.num_voxel_in_one_ensemble, :].T
        bold_y = 0.02 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
        bold_y = torch.from_numpy(bold_y.astype(np.float32)).cuda()

        w_save = [self.torch_2_numpy(w, is_cuda=True)]
        print("\n                 BEGIN DA               \n")
        self.hp_star = self.hp
        print("self.hp_strar.dtype", self.hp_star.dtype)
        for t in range(self.T - 1):
            print("PROCESSING || %d" % t)
            bold_y_t = bold_y[t].reshape([1, self.num_voxel_in_one_ensemble])
            self.hp_log = w[:, :, :self.hp_num]  # + (self.hp_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num))).type_as(w)
            print("self.hp.dtype", self.hp.dtype)
            print("self.hp_log.dtype", self.hp_log.dtype)
            print("self.hp_log", self.hp_log.min().item(), self.hp_log.max().item())
            self.hp = self.sigmoid(self.hp_log, self.low_bound, self.up_bound)
            self.hp = 2 * self.hp_star / 3 + 1 / 3 * self.hp
            print("self.hp.dtype", self.hp.dtype)
            self.hp_star = self.hp
            print("Hp, eg: ", self.hp[0, 0, :2].data)
            self.update(self.hp[:,:,0].reshape(-1), self.hp[:,:,1].reshape(-1))

            w_hat = self.evolve()
            w_hat[:, :, -5:] += (self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(
                self.ensembles, self.num_voxel_in_one_ensemble, 5))).type_as(w_hat)

            w = self.filter(w_hat, bold_y_t, rate=0.5)
            self.bold.state_update(
                w[:, :self.num_voxel_in_one_ensemble, (self.hp_num + 1):(self.hp_num + 5)])
            w_save.append(self.torch_2_numpy(w_hat, is_cuda=True))

        print("\n                 END DA               \n")
        np.save(os.path.join(write_path, "W.npy"), w_save)
        del w_hat, w
        path = write_path + '/show/'
        os.makedirs(path, exist_ok=True)

        w_save = np.array(w_save)
        self.show_bold(w_save, self.torch_2_numpy(bold_y, is_cuda=True), self.T, path, 10)  #!!!
        hp_save = self.sigmoid(
            w_save[:, :, :, :self.hp_num].reshape(self.T * self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.torch_2_numpy(self.low_bound), self.torch_2_numpy(self.up_bound))
        hp = hp_save
        np.save(os.path.join(write_path, "hp.npy"), hp)
        self.show_hp(
            hp_save.reshape(self.T, self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.T, path, 10, self.hp_num)
        self.block.shutdown()
        print("\n\n Totally cost time:\t", time.time() - total_start)
        print("=================================================\n")


if __name__ == '__main__':
    block_dict = {"ip": "10.5.4.1:50051",
                  "block_path": "./",
                  "noise_rate": 0.007,
                  "delta_t": 0.1,
                  "print_stat": False,
                  "froce_rebase": True}
    bold_dict = {"epsilon": 200,
                 "tao_s": 0.8,
                 "tao_f": 0.4,
                 "tao_0": 1,
                 "alpha": 0.2,
                 "E_0": 0.8,
                 "V_0": 0.02}

    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--print_stat", type=bool, default=False)
    parser.add_argument("--force_rebase", type=bool, default=True)
    parser.add_argument("--block_path", type=str,
                        default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/single")
    parser.add_argument("--write_path", type=str,
                        default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/")
    parser.add_argument("--T", type=int, default=450)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hp_sigma", type=float, default=0.1)
    parser.add_argument("--bold_sigma", type=float, default=1e-6)
    parser.add_argument("--ensembles", type=int, default=100)

    args = parser.parse_args()
    block_dict.update(
        {"ip": args.ip, "block_path": args.block_path, "noise_rate": args.noise_rate, "print_stat": args.print_stat,
         "force_rebase": args.force_rebase})
    whole_brain_voxel_info_path = "/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/jianfeng_normal/A1_1_DTI_voxel_structure_data_jianfeng.mat"

    da_rest = DA_Rest_Whole_Brain(block_dict, bold_dict, whole_brain_voxel_info_path, steps=args.steps,
                                  ensembles=args.ensembles, time=args.T, hp_sigma=args.hp_sigma,
                                  bold_sigma=args.bold_sigma)
    real_parameter = np.array([0.75, 0.25], dtype=np.float32)
    para_ind = np.array([10, 11, 12, 13], dtype=np.int64)
    os.makedirs(args.write_path, exist_ok=True)
    da_rest.run(real_parameter, para_ind, bold_path=whole_brain_voxel_info_path, write_path=args.write_path)