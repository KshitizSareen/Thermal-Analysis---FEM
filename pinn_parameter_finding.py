import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import a Fully Connected Neural Network (FCN) and the exact solution function
from pinn import FCN, exact_solution

if __name__ == "__main__":

    # Set the random seed for reproducibility
    torch.manual_seed(123)

    # True damping and natural frequency values
    d, w0 = 2, 20
    print(f"True value of mu: {2*d}")  # mu = 4 for comparison later

    # Generate 40 random observation points in time from the interval [0, 1]
    t_obs = torch.rand(40).view(-1, 1)

    # Get the exact solution at those time points and add Gaussian noise
    u_obs = exact_solution(d, w0, t_obs) + 0.04 * torch.randn_like(t_obs)

    # Create a neural network PINN: input dimension = 1 (t), output = 1 (u(t)), 32 hidden units, 3 layers
    pinn = FCN(1, 1, 32, 3)

    # Define 30 physics points in time for enforcing the differential equation
    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

    # Known constant: k = w0^2
    _, k = 2 * d, w0**2

    # Initialize μ (mu) as a learnable parameter — initialized to zero
    mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    # Track mu over epochs to visualize convergence
    mus = []

    # Optimizer: jointly optimize the network parameters and mu
    optimizer = torch.optim.Adam(list(pinn.parameters()) + [mu], lr=1e-3)

    # Training loop
    for i in range(25000):
        optimizer.zero_grad()

        # ----- Physics-based loss -----
        # Predict u(t) at physics-enforcing points
        u = pinn(t_physics)

        # Compute ∂u/∂t using autograd
        dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]

        # Compute ∂²u/∂t² using autograd
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]

        # Residual of the differential equation: u'' + μ u' + k u = 0
        loss1 = torch.mean((d2udt2 + mu * dudt + k * u)**2)

        # ----- Data fitting loss -----
        # Predict u(t) at observation points
        u = pinn(t_obs)

        # Loss between noisy observed data and model output
        loss2 = torch.mean((u - u_obs)**2)

        # Total loss: weighted sum of physics loss and data loss
        lambda1 = 1e4
        loss = loss1 + lambda1 * loss2

        # Backpropagation
        loss.backward()

        # Optimizer step: update weights and mu
        optimizer.step()

        # Save current mu value for plotting
        mus.append(mu.item())

    # Plot the progression of the learned mu value over training epochs
    plt.figure(figsize=(8, 4))
    plt.plot(mus, label='Learned μ over epochs')
    plt.axhline(2 * d, color='red', linestyle='--', label='True μ = 4')
    plt.xlabel('Epoch')
    plt.ylabel('μ')
    plt.title('Convergence of Learned μ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()