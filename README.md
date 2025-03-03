# Sampling-Based CBF Method for Safe Navigation

This method explores the integration of sampling-based techniques with Control Barrier Functions (CBF) to ensure safe navigation.

## Sampling-Based Optimization

The sampling-based method iteratively optimizes a simple policy modeled as a time-independent Gaussian distribution with parameters **φ_t**, which represent the sequence of means **μ_t** and covariances **Σ_t** at each time step.

At each iteration, the optimization follows these steps:
1. Sample a batch of **N** control sequences, **u**, from the current distribution.
2. Roll out the approximate dynamics function using the sampled controls to obtain a batch of corresponding states **x** and costs **c**.

## Novel Cost Function Based on Angular Relations

A novel loss function is introduced to incorporate the angular relationship between obstacles and the goal:

```
C(θ₁, θ₂) = α₁(1 - σ(2(x - 90)) ||90 - x||) + α₂ θ₂
```

where:
- **θ₁ ∈ [0, 180°]** represents the angle between the obstacle and the vector **xₜ₊₁ - xₜ**.
- **θ₂ ∈ [0, 180°]** represents the angle between the goal and the vector **xₜ₊₁ - xₜ**.
- **α₁** and **α₂** are positive scalars.

## Updating Mean and Covariance

The mean and covariance of the control distribution are updated using a sample-based gradient:

```
μ_t = (1 - α_μ)μ_{t-1} + α_μ (∑ w_i u_t) / (∑ w_i)
Σ_t = (1 - α_σ)Σ_{t-1} + α_σ (∑ w_i (u_t - μ_t)(u_t - μ_t)^T) / (∑ w_i)
```

where:
- **α_μ** and **α_σ** are step sizes regulating updates relative to the previous values.
- **w_i** is computed as:

```
w_i = exp(-C(α₁, α₂) / β)
```

To enhance diversity in the sampling process, noise is added to the covariance matrix. The best control input **uₜ₊₁** is then selected as **μₜ₊₁**.

## Ensuring Safe Navigation with CBF

To prevent collisions with obstacles, a **Control Barrier Function (CBF)** is employed:

```
∂h/∂x (f(x) + g(x) u) + α(h(x)) ≥ 0
```

where:
- **h(x)** represents the safe set, ensuring it remains non-negative.
- The function **h(x)** can be formulated as a distance function to an obstacle.
- Only the closest obstacle is considered rather than all obstacles in the region.

## Example: Single Integrator System

As a simple demonstration, consider a **single integrator system** where the method is applied to navigate safely while optimizing the control policy.

![CBF Navigation](https://github.com/user-attachments/assets/3e6ee143-27fd-4662-875d-1e01127c3b15)

## Reference:
1. Bhardwaj, Mohak, et al. "Storm: An integrated framework for fast joint-space model-predictive control for reactive manipulation." Conference on Robot Learning. PMLR, 2022.








