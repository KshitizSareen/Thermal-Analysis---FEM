import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict


def initialize_temp_dict(num_div: int, initial_temp: float) -> Dict[int, float]:
    return {i: initial_temp for i in range(num_div + 1)}


def compute_element_matrices(length: float) -> (np.ndarray, np.ndarray):
    """Return local stiffness (K) and mass (C) matrices for 1D linear element."""
    # Stiffness matrix (thermal conductivity part)
    K_local = np.array([[1, -1], [-1, 1]]) / length

    # Mass matrix (specific heat × density part)
    C_local = np.array([[2, 1], [1, 2]]) * (length / 6)

    return K_local, C_local


def assemble_global_matrices(num_div: int, k: float, rho: float, c: float, length: float) -> (np.ndarray, np.ndarray):
    n_nodes = num_div + 1
    K_global = np.zeros((n_nodes, n_nodes))
    C_global = np.zeros((n_nodes, n_nodes))

    K_local, C_local = compute_element_matrices(length)

    for i in range(num_div):
        nodes = [i, i + 1]
        for a in range(2):
            for b in range(2):
                K_global[nodes[a], nodes[b]] += k * K_local[a, b]
                C_global[nodes[a], nodes[b]] += rho * c * C_local[a, b]

    return K_global, C_global


def apply_heat_source(Q: float, num_div: int, length: float) -> np.ndarray:
    """Distribute source term Q uniformly across the domain."""
    n_nodes = num_div + 1
    F = np.zeros(n_nodes)
    for i in range(num_div):
        val = Q * length / 2
        F[i] += val
        F[i + 1] += val
    return F


def compute_T_1d(timestep: float, duration: float, initial_temp: float,
                 num_div: int, length: float, Q: float,
                 thermal_conductivity: float, specific_heat_capacity: float, density: float):
    
    # Initialize temperature vector
    T_dict = initialize_temp_dict(num_div, initial_temp)
    T_vec = np.array([T_dict[i] for i in range(num_div + 1)])

    dx = length / num_div
    K = assemble_global_matrices(num_div, thermal_conductivity, density, specific_heat_capacity, dx)[0]
    C = assemble_global_matrices(num_div, thermal_conductivity, density, specific_heat_capacity, dx)[1]
    F = apply_heat_source(Q, num_div, dx)

    A = K + (1 / timestep) * C

    headers = ["Timestamp", "Q"] + [f"{round(i*dx, 3)}" for i in range(num_div + 1)]
    with open("temperature_output_1d.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(np.linspace(0, length, num_div + 1), T_vec, color='red')
        ax.set_ylim(initial_temp, initial_temp + 100)
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("1D Heat Conduction")
        plt.grid()

        t = 0
        while t <= duration:
            D = F + (1 / timestep) * (C @ T_vec)
            T_vec = np.linalg.solve(A, D)

            # Write to CSV
            row = [round(t, 2), Q] + list(T_vec)
            writer.writerow(row)

            # Live update plot
            line.set_ydata(T_vec)
            ax.set_title(f"Temperature at t = {round(t, 2)} s")
            plt.pause(0.05)
            t += timestep

        plt.ioff()
        plt.show()

compute_T_1d(
    timestep=0.1,
    duration=10,
    initial_temp=21.23,
    num_div=100,
    length=10.0,
    Q=2.192,
    thermal_conductivity=10,
    specific_heat_capacity=0.96,
    density=1.68
)