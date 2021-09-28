""" Quick demo: Gravity Model"""

import numpy as np

import gravity_model as gm

# just to increase readability of output
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

if __name__ == "__main__":
    gm.set_seed(123)
    traffic_matrix = gm.get_traffic_matrix(n=10)
    print(traffic_matrix)
    print(f"sum TM-1 {np.sum(traffic_matrix)}")

    gm.set_seed(123)
    traffic_matrix2 = gm.get_traffic_matrix(n=10, fixed_total=1000)
    print(f"sum TM-2 {np.sum(traffic_matrix2)}")
