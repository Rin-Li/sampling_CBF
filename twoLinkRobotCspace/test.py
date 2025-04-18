import numpy as np
import matplotlib.pyplot as plt

    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def L2(x_deg):
    return 2000 * sigmoid((x_deg - 90) / 1.0) * (x_deg - 90)

def L1(x_deg):
    return 1000 * x_deg



plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, 180, 1), L2(np.arange(0, 180, 1)), label='L2 Cost', color='blue')
plt.plot(np.arange(0, 180, 1), L1(np.arange(0, 180, 1)), label='L1 Cost', color='red')
plt.title('Cost Functions')
plt.xlabel('Angle (degrees)')
plt.ylabel('Cost')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
