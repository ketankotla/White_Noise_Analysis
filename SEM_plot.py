import pandas
import seaborn
import matplotlib.pylab as plt
import scipy

parent_dir = '/Users/kwass/Desktop/InVivoAnalysis/Kir2.10Mi1-ATP2+RGECO/'
seaborn.set_style('ticks')

# === Plot average ATP2 responses over time, by fly ===
DF = pandas.read_csv(parent_dir + 'measurements/graysteps-axon-ATP2-byfly.csv')
DF['stim_time'] = DF['stim_time'] - 0.2

plt.figure(figsize=(3, 3))
seaborn.lineplot(DF, x='stim_time', y='DFF', hue='genotype', errorbar='se', legend=True)
seaborn.despine()
plt.xlabel('seconds', fontsize=16)
plt.ylabel('\u0394F/F', fontsize=16)
plt.savefig(parent_dir + 'plots/average-graysteps-axons-ATP2-byfly.png', dpi=300, bbox_inches='tight')
plt.clf()

# === Plot average RGECO responses over time, by fly ===
DF = pandas.read_csv(parent_dir + 'measurements/graysteps-axon-RGECO-byfly.csv')
DF['stim_time'] = DF['stim_time'] - 0.2

plt.figure(figsize=(3, 3))
seaborn.lineplot(DF, x='stim_time', y='DFF', hue='genotype', errorbar='se', legend=True)
seaborn.despine()
plt.xlabel('seconds', fontsize=16)
plt.ylabel('\u0394F/F', fontsize=16)
plt.savefig(parent_dir + 'plots/average-graysteps-axons-RGECO-byfly.png', dpi=300, bbox_inches='tight')
plt.clf()

# === Optional peak plot code (commented) ===

# # DF_peak = DF[(DF['stim_time'] > 0.37) & (DF['stim_time'] < 0.43)]
# # plt.figure(figsize=(1.5, 3))
# # seaborn.boxplot(data=DF_peak, x='genotype', y='DFF', hue='genotype', showfliers=False)
# # seaborn.swarmplot(data=DF_peak, x='genotype', y='DFF', facecolor='white', edgecolor='black', linewidth=0.5, size=6)
# # seaborn.despine()
# # plt.xlabel('genotype', fontsize=16)
# # plt.ylabel('\u0394F/F (0.4s)', fontsize=16)
# # plt.yticks([0, 0.2, 0.4], fontsize=12)
# # plt.savefig(parent_dir + 'plots/peak-500msFFF-axons-RGECO-byfly.png', dpi=300, bbox_inches='tight')
# # plt.clf()
# # control_peaks = DF_peak[DF_peak['genotype'] == 'control']['DFF'].values
# # KD_peaks = DF_peak[DF_peak['genotype'] == 'MiltonKD']['DFF'].values
# # print(scipy.stats.mannwhitneyu(control_peaks, KD_peaks))
