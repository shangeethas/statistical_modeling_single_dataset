#libraries
import numpy as np
import matplotlib.pyplot as plt 

random_seed = 12345 # put any number you want
np.random.seed(random_seed)
number_points = 100
x = np.random.random(number_points)

# ground truth distribution (say)
m = 3
b = 5

# artificial scatter
sigma_y = 0.2 * np.random.randn(number_points)
y = m * x + b + sigma_y

#plot
z = np.polyfit(x, y, 1)
xp = np.linspace(0, 1, 100)
y_fit = z[0]*xp + z[1]
plt.plot(x,y,'.',label='Data')
plt.plot(xp,y_fit,'-', label='Best Fit m={:.2f},b={:.2f}'.format(z[0],z[1]))
plt.legend()
plt.savefig('out.png', dpi=128)
plt.close()
print('Best fit slope : ', z[0])
print('Best fit intercept : ', z[1])
print(z)
