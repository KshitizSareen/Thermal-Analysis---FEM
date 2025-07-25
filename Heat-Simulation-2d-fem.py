from collections import defaultdict
from typing import Tuple, Dict, List
from enum import Enum
import numpy as np
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def compute_area_of_triangle(x1: float, x2: float, x3: float,
                             y1: float, y2: float, y3: float) -> float:
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def formulate_k_values(x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float) -> Tuple[float, float, float, float, float, float]:
    return (
        y2 - y3, y3 - y1, y1 - y2,  
        x3 - x2, x1 - x3, x2 - x1   
    )


def formulate_k_matrix(x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float) -> List[List[float]]:
    b1, b2, b3, c1, c2, c3 = formulate_k_values(x1, x2, x3, y1, y2, y3)
    area_inv = 1 / (4 * compute_area_of_triangle(x1, x2, x3, y1, y2, y3))

    b = [b1, b2, b3]
    c = [c1, c2, c3]

    return [
        [area_inv * (b[i] * b[j] + c[i] * c[j]) for j in range(3)]
        for i in range(3)
    ]


def formulate_C_matrix(x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float) -> List[List[float]]:
    area = compute_area_of_triangle(x1, x2, x3, y1, y2, y3)
    return [
        [2 * area if i == j else area for j in range(3)]
        for i in range(3)
    ]

def formulate_Q_Matrix(x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float,
                       Q: float) -> List[float]:
    A = compute_area_of_triangle(x1, x2, x3, y1, y2, y3)
    q_value = (Q * A) / 3
    return [q_value] * 3


def formulate_F_matrix(x1: float, x2: float, x3: float,
                        y1: float, y2: float, y3: float,
                        Q: float):
                        Q_Matrix = formulate_Q_Matrix(x1,x2,x3,y1,y2,y3,Q)
                        return Q_Matrix
    

def accumulate_matrix(global_dict: Dict[Tuple[int, int], float],
                      local_nodes: List[int],
                      local_matrix: List[List[float]]) -> None:
    for i, row in enumerate(local_nodes):
        for j, col in enumerate(local_nodes):
            global_dict[(row, col)] += local_matrix[i][j]


def _initialize_temp_dict(num_div_x: int, num_div_y: int, initialTemp: float) -> Dict[int, float]:
    base_index = num_div_x + 1
    TempDict = defaultdict(float)

    for i in range(num_div_y + 1):  # Include top edge
        for j in range(num_div_x + 1):  # Include right edge
            node_id = base_index * i + j
            TempDict[node_id] = initialTemp

    return TempDict


def _process_triangle(
                      x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                      T1: int, T2: int, T3: int,
                      Q: float,
                      KDict: Dict[Tuple[int, int], float],
                      CDict: Dict[Tuple[int, int], float],
                      FDict: Dict[int, float]) -> None:
    
    T_array = [T1, T2, T3]

    K = formulate_k_matrix(x1, x2, x3, y1, y2, y3)
    C = formulate_C_matrix(x1, x2, x3, y1, y2, y3)
    F = formulate_F_matrix(
        x1, x2, x3, y1, y2, y3, Q
    )

    accumulate_matrix(KDict, T_array, K)
    accumulate_matrix(CDict, T_array, C)
    FDict[T1] += F[0]
    FDict[T2] += F[1]
    FDict[T3] += F[2]


def _create_global_matrix(matrix_dict: Dict[Tuple[int, int], float], num_div_x: int, num_div_y: int) -> np.ndarray:
    size = (num_div_x + 1) * (num_div_y + 1)
    matrix = np.zeros((size, size), dtype=float)
    
    for (i, j), value in matrix_dict.items():
        matrix[i][j] = value
    
    return matrix


def create_global_K_matrix(KDict: Dict[Tuple[int, int], float],
                           num_div_x: int, num_div_y: int,
                           thermal_conductivity: float) -> np.ndarray:
    local_K = _create_global_matrix(KDict, num_div_x, num_div_y)
    return thermal_conductivity  * local_K


def create_global_C_matrix(CDict: Dict[Tuple[int, int], float],
                           num_div_x: int, num_div_y: int,
                           specific_heat_capacity: float,
                           density: float) -> np.ndarray:
    local_C = _create_global_matrix(CDict, num_div_x, num_div_y)
    return (density * specific_heat_capacity  / 12) * local_C

def _create_global_vector(data_dict: Dict[int, float], num_div_x: int, num_div_y: int) -> np.ndarray:
    size = (num_div_x + 1) * (num_div_y + 1)
    vector = np.zeros(size, dtype=float)
    for i in range(size):
        vector[i] = data_dict.get(i, 0.0)
    return vector


def create_global_F_matrix(FDict: Dict[int, float], num_div_x: int, num_div_y: int) -> np.ndarray:
    return _create_global_vector(FDict, num_div_x, num_div_y)


def create_global_T_matrix(TDict: Dict[int, float], num_div_x: int, num_div_y: int) -> np.ndarray:
    return _create_global_vector(TDict, num_div_x, num_div_y)

def formulate_A_matrix(K,C,timestep):
    return K + ((1/timestep)*C)

def formulate_D_matrix(F,C,timestep,T):
    return F+((1/timestep)*np.dot(C,T))
    


def generate_temp_points(width: float, height: float,
                         num_div_x: int, num_div_y: int,
                         TempDict: Dict[int,float], Q: float,timestep: float,
                         thermal_conductivity: float,specific_heat_capacity: float,density: float) -> np.ndarray:

    x_interval = width / num_div_x
    y_interval = height / num_div_y
    base_index = num_div_x + 1

    KDict = defaultdict(float)
    CDict = defaultdict(float)
    FDict = defaultdict(float)

    for i in range(num_div_y):
        for j in range(num_div_x):
            # Lower triangle
            T1 =  base_index * (i + 1) + j
            T2 = base_index * i + j
            T3 = base_index * (i + 1) + (j + 1)

            x1, y1 = j * x_interval, (i + 1) * y_interval
            x2, y2 = j * x_interval, i * y_interval
            x3, y3 = (j + 1) * x_interval, (i + 1) * y_interval

            _process_triangle(x1, y1, x2, y2, x3, y3,
                              T1, T2, T3, Q,  KDict, CDict, FDict)

            # Upper triangle
            T1 = base_index * i + (j + 1)
            T2 = base_index * i + j
            T3 = base_index * (i + 1) + (j + 1)

            x1, y1 = (j + 1) * x_interval, i * y_interval
            x2, y2 = j * x_interval, i * y_interval
            x3, y3 = (j + 1) * x_interval, (i + 1) * y_interval

            _process_triangle(x1, y1, x2, y2, x3, y3,
                              T1, T2, T3, Q,  KDict, CDict, FDict)
    
    K = create_global_K_matrix(KDict,num_div_x,num_div_y,thermal_conductivity)
    C = create_global_C_matrix(CDict,num_div_x,num_div_y,specific_heat_capacity,density)
    F = create_global_F_matrix(FDict,num_div_x,num_div_y)
    T = create_global_T_matrix(TempDict,num_div_x,num_div_y)

    A = formulate_A_matrix(K,C,timestep)
    D = formulate_D_matrix(F,C,timestep,T)



    return np.linalg.solve(A, D)


def convertTempDictToT(TempDict: Dict[int, float], num_div_y: int, num_div_x: int) -> np.ndarray:
    T = np.zeros((num_div_y + 1, num_div_x + 1))
    base_index = num_div_x + 1
    for i in range(num_div_y + 1):
        for j in range(num_div_x + 1):
            key = base_index * i + j
            T[i][j] = TempDict[key]
    return T


def compute_T(timestep: float, duration: float, initialTemp: float,
              num_div_x: int, num_div_y: int, width: float,
              height: float,  Q: float,
              thermal_conductivity: float, specific_heat_capacity: float, density: float):

    # Initialize temperature dictionary
    TempDict = _initialize_temp_dict(num_div_x, num_div_y, initialTemp)

    # Calculate real coordinates of grid points
    x_interval = width / num_div_x
    y_interval = height / num_div_y
    point_coords = [(round(j * x_interval, 5), round(i * y_interval, 5))
                    for i in range(num_div_y + 1) for j in range(num_div_x + 1)]

    num_rows = num_div_y + 1
    num_cols = num_div_x + 1

    # Setup CSV output
    csv_filename = "temperature_output.csv"
    headers = ["Timestamp", "Q"] + [f"({x},{y})" for (x, y) in point_coords]
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        # Set up the live plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        matrix = np.full((num_rows, num_cols), initialTemp)
        img = ax.imshow(matrix, cmap='hot', origin='lower', aspect='auto', vmin=initialTemp, vmax=initialTemp + 100)
        cbar = plt.colorbar(img, ax=ax, label="Temperature")
        ax.set_title("Temperature Evolution")
        ax.set_xlabel("X divisions")
        ax.set_ylabel("Y divisions")
        plt.tight_layout()
        plt.show()

        t = 0
        while t <= duration:
            # Compute new temperature
            T = generate_temp_points(width, height, num_div_x, num_div_y, TempDict,
                                     Q, timestep, thermal_conductivity, specific_heat_capacity, density)

            # Update TempDict
            for j in range(len(T)):
                TempDict[j] = T[j]

            # Write to CSV
            row = {"Timestamp": round(t, 3), "Q": Q}
            for idx, (x, y) in enumerate(point_coords):
                row[f"({x},{y})"] = TempDict[idx]
            writer.writerow(row)

            # Reshape T into 2D matrix and update the heatmap
            T_matrix = np.array(T).reshape((num_rows, num_cols))
            img.set_data(T_matrix)
            ax.set_title(f"Temperature at t = {round(t, 2)}s")
            plt.pause(0.1)  # Pause to refresh plot

            t += timestep

        plt.ioff()  # Turn off interactive mode
        plt.show()





compute_T(
    timestep=0.1,      # seconds
    duration=10,       # seconds
    initialTemp=21.23,     # °C
    num_div_x=10,
    num_div_y=10,
    width=1,          # cm
    height=1,         # cm     
    Q=2.192,            # W/cm³ - much smaller in CGS
    thermal_conductivity=10,  # W/(cm·°C)
    specific_heat_capacity=0.96,  # J/(g·°C)
    density=1.68        # g/cm³
)