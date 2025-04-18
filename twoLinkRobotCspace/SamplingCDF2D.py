import math
import torch
import cvxpy as cp
from torch.distributions import MultivariateNormal
from cdf import CDF2D
from primitives2D_torch import Circle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as mplCircle

PI = math.pi

dt = 0.1                   # scalar
H = 1                      # prediction steps
n_samples = 1000           # number of trajectory samples
T = 50                     # total time horizon
beta = 1.0                 # MPPI weight



class SamplingCBF:
    def __init__(self, start: torch.Tensor, goal: torch.Tensor, obs_list, gamma: float):
        # ensure start/goal are float
        self.start = start.float()                            # [1,2]
        self.goal = goal.float()                              # [1,2]
        self.obs_list = obs_list
        self.gamma = gamma
        self.cdf = CDF2D(device=start.device)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x: [1,2], u: [2]
        return x + u*dt                                   # [1,2]

    def cost(self, x_ref: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x_ref: [2], x: [1,2] or [2], u: [2]
        x_vec = x.clone().detach().view(-1).requires_grad_(True)  # [2]
        h_val, h_dot = self.cdf.calculate_cdf(
            x_vec.unsqueeze(0).float(), self.obs_list,
            method='online_computation', return_grad=True
        )                                                 # h_val:[1], h_dot:[1,2]
        h_dot_vec = h_dot.squeeze(0)                      # [2]
        obs_normal = h_dot_vec / h_dot_vec.norm()         # [2]
        x_next = x_vec + u                           # [2]

        def L1(deg): 
            return 1000 * deg
        
        def L2(deg): 
            if deg < 90:
                return 0
            else:
                return 2000 * (deg - 90)

        move = x_next - x_vec                             # [2]
        ref = x_ref - x_vec                               # [2]
        cos1 = torch.clamp(move.dot(ref) / (move.norm() * ref.norm()), -1.0, 1.0)
        theta1 = torch.acos(cos1) * 180.0 / PI            # scalar
        cost_val = L1(theta1)

        if obs_normal.norm() > 1e-6 and h_val.item() < 1:
            cos2 = torch.clamp(move.dot(obs_normal) / (move.norm() * obs_normal.norm()), -1.0, 1.0)
            theta2 = torch.acos(cos2) * 180.0 / PI        # scalar
            cost_val = cost_val + L2(theta2)

        return cost_val                                   # scalar

    def mppi(self, mean: torch.Tensor, cov: torch.Tensor,
             n_samples: int = n_samples, H: int = H) -> torch.Tensor:
        # mean:[H,2], cov:[H,2,2]
        device = mean.device
        u_all = torch.zeros(n_samples, H, 2, dtype=torch.float32, device=device)  # [n_samples,H,2]
        for t in range(H):
            dist = MultivariateNormal(mean[t], covariance_matrix=cov[t])
            u_all[:, t, :] = dist.sample((n_samples,))        # [n_samples,2]
        return u_all                                        # [n_samples,H,2]

    def update_mean_cov(
        self,
        mu_prev: torch.Tensor,
        x: torch.Tensor,
        Sigma_prev: torch.Tensor,
        u_all: torch.Tensor,
        x_ref: torch.Tensor,
        beta: float = beta,
        alpha_mean: float = 0.2,
        alpha_cov: float = 0.2,
        reg: float = 0.01
    ):
        # mu_prev: [H,2], Sigma_prev: [H,2,2], u_all: [n_samples,H,2]
        n_samples, H, _ = u_all.shape
        device = mu_prev.device

        w_cost = torch.zeros(n_samples, dtype=torch.float32, device=device)
        for i in range(n_samples):
            w_cost[i] = self.cost(x_ref, x, u_all[i, 0])

        w = torch.exp(-beta * (w_cost - w_cost.min()))      # [n_samples]
        w_sum = w.sum()

        weighted = (w.view(n_samples,1,1) * u_all).sum(dim=0) / w_sum  # [H,2]
        mu_new = (1 - alpha_mean) * mu_prev + alpha_mean * weighted    # [H,2]

        Sigma_new = torch.zeros_like(Sigma_prev, device=device)
        diff = u_all - mu_new.unsqueeze(0)                            # [n_samples,H,2]
        for h in range(H):
            dh = diff[:, h, :]                                       # [n_samples,2]
            cov_w = torch.einsum('i,ij,ik->jk', w, dh, dh) / w_sum  # [2,2]
            S = (1 - alpha_cov) * Sigma_prev[h] + alpha_cov * cov_w
            S = S + reg * torch.eye(2, dtype=torch.float32, device=device)
            diag = torch.clamp(torch.diag(S), min=reg)
            S[0,0], S[1,1] = diag[0], diag[1]
            Sigma_new[h] = S                                      # [2,2]

        best_u = mu_new[0]                                         # [2]
        return best_u, mu_new, Sigma_new

    def CBF(self, current_state: torch.Tensor, goal: torch.Tensor) -> np.ndarray:
        # current_state: [1,2], goal: [2]
        x = current_state.clone().detach().float().requires_grad_(True)
        h_val, h_dot = self.cdf.calculate_cdf(
            x.view(1, -1), self.obs_list,
            method='online_computation', return_grad=True
        )                                                 # [1],[1,2]
          # [2]

        u_goal = goal.detach().cpu().numpy().flatten()    # [2]
        h_dot_np = h_dot.detach().cpu().numpy().reshape(1, -1)  # [1,2]
        x_np = x.detach().cpu().numpy().flatten()         # [2]
        u = cp.Variable(2)
        obj = cp.Minimize(cp.norm(u - u_goal))
        cons = [h_dot_np @ u + self.gamma * h_val.item() >= 0,
                x_np[0] + u[0] >= -PI, x_np[0] + u[0] <= PI,
                x_np[1] + u[1] >= -PI, x_np[1] + u[1] <= PI]
        prob = cp.Problem(obj, cons)
        prob.solve()
        return u.value if u.value is not None else np.zeros(2)    # [2]

    def optimize(self):
        mean = torch.zeros(H, 2, dtype=torch.float32, device=self.start.device)         # [H,2]
        cov = torch.eye(2, dtype=torch.float32, device=self.start.device).repeat(H,1,1)  # [H,2,2]
        x_traj = [self.start.cpu().numpy().flatten()]
        u_all_every_step = []                                      # list of [n_samples,H,2]


        for _ in range(int(T / dt)):
            all_u = self.mppi(mean, cov)
            u_all_every_step.append(all_u.cpu().numpy())
            x_current = torch.tensor(x_traj[-1], dtype=torch.float32, device=self.start.device)
            best_u, mean, cov = self.update_mean_cov(mean, x_current, cov, all_u, self.goal)
            u_cbf = self.CBF(x_current.unsqueeze(0), best_u)                   # [2]
            u_cbf_t = torch.tensor(u_cbf, dtype=torch.float32, device=self.start.device)
            x_next = self.dynamics(x_current, u_cbf_t)            # [1,2]
            x_traj.append(x_next.cpu().numpy().flatten())
            print("x_next: ", x_next.cpu().numpy().flatten())
       
            if (x_next - self.goal).norm() < 0.1:
                break

        return x_traj, u_all_every_step


def plot_2link_with_obstacle(states,
                              link_lengths=(2, 2),
                              obstacle_center=(2.5, 2.5),
                              obstacle_radius=0.5,
                              show_traj=True):
    l1, l2 = link_lengths
    traj = np.array(states)                                   # [N,2]

    plt.figure(figsize=(8, 8))                                 
    ax = plt.gca()
    obs_circle = mplCircle(obstacle_center, obstacle_radius,
                           color='gray', alpha=0.4)
    ax.add_patch(obs_circle)

    for q in traj:
        theta1, theta2 = q
        j1 = np.array([l1*math.cos(theta1), l1*math.sin(theta1)])
        j2 = j1 + np.array([l2*math.cos(theta1+theta2),
                            l2*math.sin(theta1+theta2)])
        plt.plot([0, j1[0], j2[0]], [0, j1[1], j2[1]], 'o-', alpha=0.3)

    if show_traj:
        plt.plot(traj[:,0], traj[:,1], 'r.-')
    plt.axis('equal'); plt.grid(); plt.show()


def main():
    obs = Circle(torch.tensor([2.5, 2.5], dtype=torch.float32), radius=0.5)
    goal = torch.tensor([0.8 * PI, 0], dtype=torch.float32)
    start = torch.tensor([[-PI/2, 0]], dtype=torch.float32)
    solver = SamplingCBF(start, goal, [obs], gamma=0.6)
    states, _ = solver.optimize()
    np.save('states.npy', states)
    plot_2link_with_obstacle(states)
    


if __name__ == '__main__':
    main()
