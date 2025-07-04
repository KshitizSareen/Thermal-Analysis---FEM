from collections import defaultdict
from typing import Tuple, Dict, List
from enum import Enum
import numpy as np
import time


class TriangleType(Enum):
    LEFT = 0
    BOTTOM_LEFT = 1
    BOTTOM = 2
    BOTTOM_RIGHT = 3
    RIGHT = 4
    TOP_RIGHT = 5
    TOP = 6
    INTERNAL = 7

def compute_area_of_triangle(x1: float, x2: float, x3: float,
                             y1: float, y2: float, y3: float) -> float:
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def formulate_k_values(x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float) -> Tuple[float, float, float, float, float, float]:
    return (
        y2 - y1, y3 - y1, y1 - y2,
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


def _conv_vector(hout: float, Tout: float, T1: float, T2: float, length: float, idx1: int, idx2: int) -> List[float]:
    """Helper to compute convection contribution over a single edge."""
    coeff1 = (hout * length / 6) * (3 * Tout - 2 * T1 - T2)
    coeff2 = (hout * length / 6) * (3 * Tout - 2 * T2 - T1)
    result = [0.0, 0.0, 0.0]
    result[idx1] = coeff1
    result[idx2] = coeff2
    return result


def formulate_B_Matrix(hout: float, Tout: float, T1: float, T2: float, T3: float,
                       triangle_type: TriangleType,
                       x1: float, x2: float, x3: float,
                       y1: float, y2: float, y3: float) -> List[float]:
    
    if triangle_type == TriangleType.INTERNAL:
        return [0.0, 0.0, 0.0]

    result = [0.0, 0.0, 0.0]

    if triangle_type in {TriangleType.LEFT, TriangleType.BOTTOM_LEFT, TriangleType.TOP_RIGHT, TriangleType.RIGHT}:
        le_y = abs(y2 - y1)
        edge_result = _conv_vector(hout, Tout, T1, T2, le_y, 0, 1)
        result = [sum(x) for x in zip(result, edge_result)]

    if triangle_type in {TriangleType.BOTTOM, TriangleType.BOTTOM_LEFT, TriangleType.TOP_RIGHT, TriangleType.TOP}:
        le_x = abs(x3 - x2)
        edge_result = _conv_vector(hout, Tout, T1, T3, le_x, 0, 2)
        result = [sum(x) for x in zip(result, edge_result)]

    return result

def formulate_F_matrix(x1: float, x2: float, x3: float,
                        y1: float, y2: float, y3: float,
                        Q: float,hout: float, Tout: float, 
                        T1: float, T2: float, T3: float,
                        triangle_type: TriangleType):
                        B_Matrix = formulate_B_Matrix(hout,Tout,T1,T2,T3,triangle_type,x1,x2,x3,y1,y2,y3)
                        Q_Matrix = formulate_Q_Matrix(x1,x2,x3,y1,y2,y3,Q)
                        return [B_Matrix[0]+Q_Matrix[0],B_Matrix[1]+Q_Matrix[1],B_Matrix[2]+Q_Matrix[2]]
    

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

def _identify_triangle_type(i: int, j: int, triangle_kind: str,
                            num_div_y: int, num_div_x: int) -> TriangleType:
    if triangle_kind == 'lower':
        if j == 0:
            if i < num_div_y - 1:
                return TriangleType.LEFT
            elif i == num_div_y - 1:
                return TriangleType.BOTTOM_LEFT
        elif i == num_div_y - 1 and j > 0:
            return TriangleType.BOTTOM
    elif triangle_kind == 'upper':
        if j == num_div_x - 1:
            if i > 0:
                return TriangleType.RIGHT
            elif i == 0:
                return TriangleType.TOP_RIGHT
        elif i == 0 and j < num_div_x - 1:
            return TriangleType.TOP
    return TriangleType.INTERNAL


def _process_triangle(i: int, j: int, triangle_kind: str,
                      num_div_y: int, num_div_x: int,
                      x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                      T1: int, T2: int, T3: int,
                      Q: float, hout: float, Tout: float,
                      KDict: Dict[Tuple[int, int], float],
                      CDict: Dict[Tuple[int, int], float],
                      FDict: Dict[int, float],
                      TDict: Dict[int, float]) -> None:
    
    T_array = [T1, T2, T3]
    triangle_type_enum = _identify_triangle_type(i, j, triangle_kind, num_div_y, num_div_x)

    K = formulate_k_matrix(x1, x2, x3, y1, y2, y3)
    C = formulate_C_matrix(x1, x2, x3, y1, y2, y3)
    F = formulate_F_matrix(
        x1, x2, x3, y1, y2, y3, Q, hout, Tout,
        TDict[T1], TDict[T2], TDict[T3], triangle_type_enum
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
                         hout: float, Tout: float,
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
            T1 = base_index * i + j
            T2 = base_index * (i + 1) + j
            T3 = base_index * (i + 1) + (j + 1)

            x1, y1 = j * x_interval, i * y_interval
            x2, y2 = j * x_interval, (i + 1) * y_interval
            x3, y3 = (j + 1) * x_interval, (i + 1) * y_interval

            _process_triangle(i, j, 'lower', num_div_y,num_div_x,x1, y1, x2, y2, x3, y3,
                              T1, T2, T3, Q, hout, Tout, KDict, CDict, FDict,TempDict)

            # Upper triangle
            T1 = base_index * i + j
            T2 = base_index * i + (j + 1)
            T3 = base_index * (i + 1) + (j + 1)

            x1, y1 = j * x_interval, i * y_interval
            x2, y2 = (j + 1) * x_interval, i * y_interval
            x3, y3 = (j + 1) * x_interval, (i + 1) * y_interval

            _process_triangle(i, j, 'upper', num_div_y,num_div_x, x1, y1, x2, y2, x3, y3,
                              T1, T2, T3, Q, hout, Tout, KDict, CDict, FDict,TempDict)
    
    K = create_global_K_matrix(KDict,num_div_x,num_div_y,thermal_conductivity)
    C = create_global_C_matrix(CDict,num_div_x,num_div_y,specific_heat_capacity,density)
    F = create_global_F_matrix(FDict,num_div_x,num_div_y)
    T = create_global_T_matrix(TempDict,num_div_x,num_div_y)

    A = formulate_A_matrix(K,C,timestep)
    D = formulate_D_matrix(F,C,timestep,T)



    return np.dot(np.linalg.inv(A),D)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict

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
              height: float, hout: float, Tout: float, Q: float,
              thermal_conductivity: float, specific_heat_capacity: float, density: float):
    
    # Initialize temperature dictionary
    TempDict = _initialize_temp_dict(num_div_x, num_div_y, initialTemp)
    steps = int(duration / timestep)

    # Set up the plot
    fig, ax = plt.subplots()
    matrix = convertTempDictToT(TempDict, num_div_y, num_div_x)
    img = ax.imshow(matrix, cmap='hot', interpolation='nearest', origin='lower', vmin=0, vmax=200)
    plt.colorbar(img, ax=ax)
    title = ax.set_title("Temperature Evolution")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Animation update function
    def update(frame):
        nonlocal TempDict
        T = generate_temp_points(width, height, num_div_x, num_div_y, hout, Tout, TempDict,
                                 Q, timestep, thermal_conductivity, specific_heat_capacity, density)
        size = (num_div_x + 1) * (num_div_y + 1)
        for j in range(size):
            TempDict[j] = T[j]
        
        matrix = convertTempDictToT(TempDict, num_div_y, num_div_x)
        img.set_data(matrix)
        title.set_text(f"Time = {frame * timestep:.1f}s")
        return [img]

    # Create animation (disable blitting)
    ani = FuncAnimation(fig, update, frames=range(steps), interval=timestep * 1000, blit=False)
    plt.show()





compute_T(1,100000,50,1,1,100,100,1,20,-1,2.87,0.4,10)