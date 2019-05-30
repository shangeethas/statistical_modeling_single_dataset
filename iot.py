#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_frequencies = np.load('Frequency.npy')
print('df_frequencies.shape : ', df_frequencies.shape)

fig, ax = plt.subplots(1, 2)
print(ax.shape)

df_auditorium_46 = np.load('Complex_S21_Auditorium_46.npy')
print('df_auditorium_46.shape : ', df_auditorium_46.shape)
print(df_auditorium_46)
quit()
for i in range(196):
    ax[0].plot(x, df_auditorium_46[0 ,:, i],lw=2, alpha=0.6)
ax[0].set_xlabel('Frequencies')
ax[0].set_ylabel('S21 Parameter')
ax[0].set_title('Auditorium 46 Environment')

df_auditorium_96 = np.load('Complex_S21_Auditorium_96.npy')
print('df_auditorium_96.shape : ', df_auditorium_96.shape)
for i in range(196):
    ax[1].plot(x, df_auditorium_96[0 ,:, i],lw=2, alpha=0.6)
ax[1].set_xlabel('Frequencies')
ax[1].set_ylabel('S21 Parameter')
ax[1].set_title('Auditorium 96 Environment')

plt.show()




















































