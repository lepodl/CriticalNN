import torch
import numpy as np

class block:
    def __init__(self, node_property, w_uij, delta_t=1, external=False, specified_src_in_weight=False, external_label=None, external_index=None, src=None, poission_external=None, noise_rate=0.01):
        # A block is a set of spliking neurals with inner full connections, we consider 4 connections:
        # AMPA, NMDA, GABAa and GABAb
        # shape note:
        #
        # N: numbers of neural cells
        # K: connections kind, = 4 (AMPA, NMDA, GABAa and GABAb)
        assert len(w_uij.shape) == 3
        N = w_uij.shape[1]
        K = w_uij.shape[0]

        self.w_uij = w_uij  # shape [K, N, N]
        self.src = src
        self.src_neuron = None
        self.iter = None
        self.poisson_external = poission_external
        self.delta_t = delta_t
        self.update_property(node_property)
        if external:
            if external_index is not None:
                self.src_neuron = torch.from_numpy(external_index)
                self.iter = 0
            elif specified_src_in_weight:
                if isinstance(self.w_uij, torch.sparse.Tensor):
                    non_src_neuron = torch.unique(self.w_uij.coalesce().indices()[1])
                else:
                    non_src_neuron = torch.unique(self.w_uij.nonzero()[:, 1])
                idx = torch.arange(node_property.shape[0], dtype=torch.int64)
                idx = idx[torch.from_numpy(~np.isin(idx.numpy(), non_src_neuron.numpy()))].contiguous()
                self.src_neuron = idx
                if self.src is not None:
                    assert self.src_neuron.shape[0] == self.src.shape[0]
                self.iter = 0
            elif external_label is not None:
                self.src_neuron = torch.where(self.sub_idx==external_label)
                self.iter = 0
            else:
                raise NotImplementedError
            self.extern_length = len(self.src_neuron[0])
            self.normal_length = N - self.extern_length
            # print("extern_idx", self.src_neuron)
            # print("extern_legnth", self.extern_length)
        if isinstance(noise_rate, np.ndarray):
            noise_rate = torch.tensor(noise_rate, dtype=torch.float32).to(self.w_uij.device)
            assert len(noise_rate) == len(torch.unique(self.sub_idx)), "noise_rate.shape incositent with sub_idx.shape"
            self.noise_rate = torch.zeros(N, dtype=torch.float32, device=self.w_uij.device)
            for i, idx in enumerate(torch.unique(self.sub_idx)):
                self.noise_rate[self.sub_idx==idx] = noise_rate[i]
        else:
            self.noise_rate = noise_rate

        self.t_ik_last = torch.zeros([N], device=self.w_uij.device) # shape [N]
        self.V_i = torch.ones([N], device=self.w_uij.device) * (self.V_th + self.V_reset)/2  # membrane potential, shape: [N]
        self.J_ui = torch.zeros([K, N], device=self.w_uij.device)  # shape [K, N]
        self.t = torch.tensor(0., device=self.w_uij.device)  # scalar

        self.update_I_syn()
        print("subblk_length:", len(torch.unique(self.sub_idx)))

    @staticmethod
    def expand(t, size):
        t = torch.tensor(t)
        shape = list(t.shape) + [1] * (len(size) - len(t.shape))
        return t.reshape(shape).expand(size)

    def update_J_ui(self, delta_t, active):
        # active shape: [N], dtype bool
        # t is a scalar
        self.J_ui = self.J_ui * torch.exp(-delta_t / self.tau_ui)
        J_ui_activate_part = self.bmm(self.w_uij, active.float()) # !!! this part can be sparse.
        self.J_ui += J_ui_activate_part
        pass

    @staticmethod
    def bmm(H, b):
        if isinstance(H, torch.sparse.Tensor):
            return torch.stack([torch.sparse.mm(H[i], b.unsqueeze(1)).squeeze(1) for i in range(4)])
        else:
            return torch.matmul(H, b.unsqueeze(0).unsqueeze(2)).squeeze(2)

    def update_I_syn(self):
        self.I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        # [K, N]            [K, N] - [K, 1]
        self.I_syn = self.I_ui.sum(dim=0)
        pass

    def update_Vi(self, delta_t):
        main_part = -self.g_Li * (self.V_i - self.V_L)
        C_diff_Vi = main_part + self.I_syn + self.I_extern_Input
        delta_Vi = delta_t / self.C * C_diff_Vi

        Vi_normal = self.V_i + delta_Vi

        # if t < self.t_ik_last + self.T_ref:
        #   V_i = V_reset
        # else:
        #   V_i = Vi_normal
        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        #print(is_not_saturated.sum())
        active = torch.ge(V_i, self.V_th)
        if self.src_neuron is not None:
            if self.src is not None:
                extern_active = self.src[:, self.iter]
                active[self.src_neuron] = extern_active
            else:
                # if excludes src_neruon, other neurons have channel noise.
                # active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < self.noise_rate) | active
                extern_active = torch.rand(self.extern_length, device=self.w_uij.device) < float(self.poisson_external)
                active[self.src_neuron] = extern_active
            self.iter += 1
            self.V_i[self.src_neuron] = torch.where(extern_active,
                                                    self.V_th[self.src_neuron],
                                                    self.V_reset[self.src_neuron])
        self.V_i = torch.min(V_i, self.V_th)
        return active

    def update_t_ik_last(self, active):
        self.t_ik_last = torch.where(active, self.t, self.t_ik_last)

    def run(self, apply_noise=True, isolated=False):
        self.t += self.delta_t
        self.active = self.update_Vi(self.delta_t)
        if apply_noise:
            if not isolated:
                new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < self.noise_rate) | self.active
            else:
                new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < self.noise_rate)
        else:
            new_active= self.active
        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        self.update_t_ik_last(self.active)

        mean_Vi = []
        sum_activate = []
        mean_Iui = []
        # mean_activate = []
        for i in torch.unique(self.sub_idx):
            # mean_Vi.append(self.V_i[self.sub_idx == i].mean())
            sum_activate.append(self.active[self.sub_idx== i].float().mean())
            mean_Iui.append(self.I_ui[:, self.sub_idx == i].float().mean(axis=1))
            # mean_activate.append(self.active[self.sub_idx == i].float().mean())

        return torch.stack(sum_activate), torch.stack(mean_Iui)

    def update_property(self, node_property):
        # update property
        # column of node_property is
        # E/I, blocked_in_stat, has_extern_Input, no_input, C, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui
        E_I, blocked_in_stat, I_extern_Input, sub_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui = \
            node_property.transpose(0, 1).split([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4])

        self.I_extern_Input = I_extern_Input.squeeze(0) # extern_input index , shape[K]
        self.V_ui = V_ui  # AMPA, NMDA, GABAa and GABAb potential, shape [K, N]
        self.tau_ui = tau_ui  # shape [K, N]
        self.g_ui = g_ui  # shape [K, N]
        self.g_Li = g_Li.squeeze(0)  # shape [N]
        self.V_L = V_L.squeeze(0)  # shape [N]
        self.C = C.squeeze(0)   # shape [N]
        self.sub_idx = sub_idx.squeeze(0) # shape [N]
        self.V_th = V_th.squeeze(0)   # shape [N]
        self.V_reset = V_reset.squeeze(0)  # shape [N]
        self.T_ref = T_ref.squeeze(0) # shape [N]
        return True

    def update_conn_weight(self, conn_idx, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        self.w_uij[conn_idx] = conn_weight
        return True

