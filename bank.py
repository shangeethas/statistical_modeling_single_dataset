# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df_bank_full = pd.read_csv('bank-full.csv', sep=';')
df = pd.read_csv('bank.csv', sep=';')
print('Bank shape : ', df.shape)
print('Bank Full Shape : ', df_bank_full.shape)
print('Headers of Bank Data : ', df.describe())
print('Headers of Bank Full Data : ', df_bank_full.describe())

# poisson
fig, ax = plt.subplots(1, 1)
xi = df_bank_full['day'].value_counts().index
yi = df_bank_full['day'].value_counts(normalize=True)
ax.vlines(xi, 0, yi, colors='b', lw=5, label='Histogram of data')

day_mu = df_bank_full['day'].mean()
day_sigma = df_bank_full['day'].std()

print('Day Mu : ', day_mu)
print('Day Sigma : ', day_sigma)

rate = day_mu
x = np.arange(0, 35, 1)
y = stats.poisson.pmf(x, rate)
plt.plot(x, y, 'o-', label='Poisson PMF')
plt.legend()
plt.savefig('day_poisson.png', dpi=128)
plt.close()

# negative binomial
fig, ax = plt.subplots(1, 1)
xn = df_bank_full['campaign'].value_counts().index
yn = df_bank_full['campaign'].value_counts(normalize=True)
ax.vlines(xn, 0, yn, colors='g', lw=1, label='Histogram of data')

campaign_mu = df_bank_full['campaign'].mean()
campaign_sigma = df_bank_full['campaign'].std()

N = df_bank_full['campaign'].value_counts().index
values = df_bank_full['campaign'].value_counts().keys().tolist()
counts = df_bank_full['campaign'].value_counts().tolist()
p = counts[0] / df_bank_full['campaign'].size
k = 3  # peak_values 1, 2 
dis = stats.nbinom.pmf(range(len(N)), k, p)
plt.plot(range(len(N)), dis, color='black', lw=1, label='Negative Binomial PMF')
plt.xlabel('No of Campaigns')
plt.ylabel('{} success with 0:{} fails'.format(k, N - 1))
plt.title('Negative Binomial Distribution, p = {}'.format(p))

plt.legend()
plt.savefig('campaign_negative_binomial.png', dpi=128)
plt.close()

# central_limit_theorem
sample_mean = []
sample_variance = []


def clt_data_population(iterations, samples_per_iteration):
    for iteration in range(iterations):
        df_subsample = df_bank_full.sample(samples_per_iteration)
        mean = df_subsample['balance'].mean()
        sample_mean.append(mean)

        variance = df_subsample['balance'].std()
        sample_variance.append(variance)

    return sample_mean, sample_variance


plt.figure(figsize=(8, 8))
plt.hist(df_bank_full['balance'], bins=50, density=True)
plt.title('Data Representation - Long Term Deposity Balance')
plt.xlabel('Amount of Long Term Deposit Balance')
plt.ylabel('Probability of Deposit Holders')

population_mean = df_bank_full['balance'].mean()
population_variance = df_bank_full['balance'].std()
plt.axvline(population_mean)
plt.axvline(population_variance)
plt.annotate('Population Mean of Balance in Bank Full Data : {:.2f}'.format(population_mean), xy=(0.9, 0.9),
             xycoords='axes fraction')
plt.annotate('Population Variance of Balance in Bank Full Data : {:.2f}'.format(population_variance), xy=(0.9, 0.9),
             xycoords='axes fraction')
plt.savefig('balance_population_clt.png', dpi=128)
plt.close()

sample_mean, sample_variance = clt_data_population(1000, 10000)
mean_sample_mean = np.mean(sample_mean)
mean_sample_variance = np.std(sample_mean)
mean_of_variances = np.mean(sample_variance)

print('Mean of Sample Means : ', mean_sample_mean)
print('Mean of Sample Variance : ', mean_sample_variance)

plt.hist(sample_mean, bins=50, density=True, histtype='step')
plt.axvline(mean_sample_mean)
plt.title('CLT Approximation - Sample Means Iterations = {:5d} Samples at each Iteration = {:5d}'.format(1000, 10000),
          fontsize=8)
plt.xlabel('Long Term Deposit Balance Means')
plt.ylabel('Probability of Mean Deposit Holders')
plt.savefig('balance_sample_mean_clt_1000_10000.png', dpi=128)
plt.close()

plt.hist(sample_variance, bins=50, density=True, histtype='step')
plt.axvline(mean_of_variances)
plt.title('CLT Approximation - Sample Variance')
plt.xlabel('Long Term Deposit Balance Variances')
plt.ylabel('Probability of Variance Deposit Holders')
plt.savefig('balance_sample_variance_clt_1000_10000.png', dpi=128)
plt.close()
