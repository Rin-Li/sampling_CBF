import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cvxpy as cp

# Parameter settings
dt = 0.1          # Time step
H = 1            # Prediction horizon
n_samples = 1000   # Number of samples
R = np.eye(2)     # Control cost matrix
Q = np.diag([1, 1])     # State cost matrix
q_ref = np.array([-2.0, -2.0])  # Reference point
gamma = 1       # Discount factor
alpha_mean = 0.2  # Mean update smoothing rate
alpha_cov = 0.2 # Covariance update smoothing rate
T = 20    # Total simulation time
beta = 1    # Temperature parameter
obs_center = np.array([[2, 2], [-2, 0], [1, -2], [4, 0], [0, 0]])
obs_radii = np.array([1, 1, 1, 1, 1])
x_start = np.array([3.5, 3.5])

def plot_simulation_result(states, start, goal, obs_center, obs_radii):
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    
    plt.figure(figsize = (6, 6))
    
    for i in range(len(obs_center)):
        circle = plt.Circle((obs_center[i][0], obs_center[i][1]), np.sqrt(obs_radii[i]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_patch(circle)
    

    plt.plot(x_vals, y_vals, '-o', label='Trajectory', alpha=0.5)
    
    plt.scatter(start[0], start[1], s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(goal[0], goal[1], s=200, color="purple", alpha=0.75, label="target position")
    
    
    plt.title('Result')
    plt.xlabel('X ')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    

def get_predicted_trajecotories(current_state, control_sequence):
    predicted_state = [current_state]

    for ctrl in control_sequence:
      next_state = dynamics(predicted_state[-1], ctrl)
      predicted_state.append(next_state.copy())
    return np.array(predicted_state)
    
  
def animate_simulation(newStates, start, goal, obs_center, obs_radii, sampled_us=[]):
    states = newStates[1:].copy()
    def update(frame, states):
        plt.gca().cla()
        
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        
        plt.gca().set_aspect('equal', adjustable='box')
        
        for i in range(len(obs_center)):
            circle = plt.Circle((obs_center[i][0], obs_center[i][1]), np.sqrt(obs_radii[i]), color='blue', fill=True, linestyle='-', linewidth=2, alpha=0.5)
            plt.gca().add_artist(circle)
        
        if len(sampled_us) > 0:
            # pred_trajs = [get_predicted_trajecotories(states[frame], sampled_us[frame][i]) for i in range(len(sampled_us))]
            num_trajs_plotted = np.minimum(50,  len(sampled_us[frame]))  # plot maximum 50 trajs per step
            for idx in range(num_trajs_plotted):
                pred_traj = get_predicted_trajecotories(states[frame], sampled_us[frame][idx])
                x_pos, y_pos = pred_traj[:, 0], pred_traj[:, 1]
                plt.plot(x_pos, y_pos, color="k", alpha=0.1)
            
        
        plt.plot([state[0] for state in states[:frame+1]], [state[1] for state in states[:frame+1]], '-o', markersize=4, alpha=0.5)
        
        plt.scatter(start[0], start[1], s=200, color="green", alpha=0.75, label="start")
        plt.scatter(goal[0], goal[1], s=200, color="purple", alpha=0.75, label="target")

        plt.title('Simulation Result for Single Integrator')
        plt.xlabel('X ')
        plt.ylabel('Y ')
        plt.grid()
        plt.legend(loc="upper left")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), interval=100, blit=False)
    anim.save('simulation.gif', writer='pillow')
    
def dynamics(x, u, dt=dt):
    return x + u * dt


def collision_check(x, obs, r):
    for i in range(len(obs)):
        if np.linalg.norm(x - obs[i]) <= r[i]:
            return True
    return False

def CBF(start, goal, obs_center, obs_radii):
    u = cp.Variable(2)
    
    u_goal = goal
    
    objective = cp.Minimize(cp.norm(u - u_goal))
    constraints = []
    for center, radii in zip(obs_center, obs_radii):
        h = cp.norm(start - center)**2 - radii**2
        h_dot = 2 * (start - center)
        constraints.append(h_dot @ u + gamma * h >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return u.value if u.value is not None else np.zeros(2)
    

def cost(x_ref, x, u, obs, r):
    """
    Cost function about angle
    """
    
    x_state = [x]
    for i in range(len(u)):
        x_state.append(dynamics(x_state[i], u[i], dt))
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def L2(x):
        return 2000 * (1 - sigmoid((x - 90)/ 0.5)) * np.sqrt((90 - x) ** 2)

    def L1(x):
        return 2000 * x
    
    cost = 0
    for i in range(len(x_state) - 1):
        theta1 = np.arccos(np.dot(x_state[i + 1] - x_state[i], x_ref - x_state[i]) / (np.linalg.norm(x_state[i + 1] - x_state[i]) * np.linalg.norm(x_ref - x_state[i])))
        for j in range(obs.shape[0]):
            
            theta2 = np.arccos(np.dot(x_state[i + 1] - x_state[i], obs[j][0] - x_state[i]) / (np.linalg.norm(x_state[i + 1] - x_state[i]) * np.linalg.norm(obs[j][0] - x_state[i])))
            theta2_deg = np.rad2deg(theta2)
            cost += L2(theta2_deg)
    
        theta1_deg = np.rad2deg(theta1)

        
        cost += L1(theta1_deg)
    return cost

def mppi(mean, cov, n_samples=n_samples, H=H):

    u_all = np.zeros((n_samples, H, 2))
    for i in range(n_samples):
        for h in range(H):
            u_all[i, h] = np.random.multivariate_normal(mean[h], cov[h])

    return u_all

def update_mean_cov(mu_prev, x, obs, obs_r, Sigma_prev, u_all, gamma=0.8, alpha_mu=0.2, alpha_sigma=0.2, reg=0.01):


    
    w_cost_all = np.zeros(u_all.shape[0])
    for i in range(u_all.shape[0]):
        w_cost = cost(q_ref, x, u_all[i], obs, obs_r)
        w_cost_all[i] = w_cost

    w = np.exp(-beta * (w_cost_all - np.min(w_cost_all)))
    print("min w: ", np.min(w_cost_all))
    print("max w: ", np.max(w_cost_all))
    

    w_sum = np.sum(w)

  
    H_len = mu_prev.shape[0]
    #mu
    mu_new = np.zeros_like(mu_prev)
    for h in range(H_len):
        weighted_sum = np.sum(w[:, None] * u_all[:, h], axis=0)
        mu_new[h] = (1 - alpha_mu) * mu_prev[h] + alpha_mu * (weighted_sum / w_sum)
    
    #Sigma
    Sigma_new = np.zeros_like(Sigma_prev)
    for h in range(H_len):
        diff = u_all[:, h] - mu_new[h]
        weighted_cov = np.zeros((2, 2))
        for i in range(u_all.shape[0]):
            weighted_cov += w[i] * np.outer(diff[i], diff[i])
        weighted_cov /= w_sum
        Sigma_new[h] = (1 - alpha_sigma) * Sigma_prev[h] + alpha_sigma * weighted_cov
        Sigma_new[h] += reg * np.eye(2)
        Sigma_new[h] = np.maximum(Sigma_new[h], 0.01 * np.eye(2))

  
    print("best_u: ", mu_new[0])
    return mu_new[0], mu_new, Sigma_new


def main():
    x_traj = [x_start]
    mean = np.zeros((H, 2))
    cov = np.tile(np.eye(2), (H, 1, 1))
    u_all_all =[]
    for t in np.arange(0, T, dt):
        all_u = mppi(mean, cov)
        u_all_all.append(all_u)
        cur_obs_center = []
        cur_obs_radii = []
        for j in range(obs_center.shape[0]):
            if(np.linalg.norm(x_traj[-1] - obs_center[j]) < obs_radii[j] + 0.5):
                cur_obs_center.append(obs_center[j])
                cur_obs_radii.append(obs_radii[j])
        cur_obs_center, cur_obs_radii = np.array(cur_obs_center), np.array(cur_obs_radii)
        best_u, mean, cov = update_mean_cov(mean, x_traj[-1], cur_obs_center, cur_obs_radii, cov, all_u, reg=1e-6)
        
        
        
        best_u_CBF = CBF(x_traj[-1], best_u, cur_obs_center, cur_obs_radii)
        x_next = dynamics(x_traj[-1], best_u_CBF)
        
        print(x_next)
        x_traj.append(x_next)
        if np.linalg.norm(x_next - q_ref) < 0.2:
            break
    
    animate_simulation(x_traj, x_start, q_ref, obs_center, obs_radii, u_all_all)

if __name__ == '__main__':
    main()

    
    
    
    
  


