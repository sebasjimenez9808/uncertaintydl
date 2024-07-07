times_by_model = {'Deep Ensemble': 767.5295906066895, 'Ensemble': 116.12449908256531, 'Hamiltonian Monte Carlo': 33.38292860984802, 'Laplace': 4.740764379501343, 'MC Dropout': 8.783745050430298, 'Reference': 299.2600076198578, 'Variational Inference': 24.915364980697632}

# mc dropout
# Accuracy: 0.6275sla
# model_mc.aleatoric_entropy.mean()
# Out[5]: tensor(0.5941)
# model_mc.aleatoric_uncertainty.mean()
# Out[6]: tensor(1.4823)
# model_mc.epistemic_uncertainty.mean()
# Out[7]: tensor(0.0456)

# bootstrap
# Accuracy: 0.654
# model_boots.aleatoric_entropy.mean()
# Out[4]: tensor(0.5696)
# model_boots.aleatoric_uncertainty.mean()
# Out[5]: tensor(0.0227)
# model_boots.epistemic_uncertainty.mean()
# Out[6]: tensor(0.0020)

# MCMC
# accuracy: 0.518
# aleatoric entropy tensor(0.6174)
# epistemic uncertainty tensor(0.0992)
# aleatoric uncertainty tensor(3.6340)

# Laplace
# accuracy: 0.656
# aleatoric entropy tensor(0.5861)
# epistemic uncertainty tensor(0.0210)
# aleatoric uncertainty tensor(0.2116)

# VI
# model_vi.aleatoric_entropy.mean()
# Out[6]: tensor(0.4897)
# model_vi.epistemic_uncertainty.mean()
# Out[7]: tensor(0.1062)
# model_vi.aleatoric_uncertainty.mean()
# Out[8]: tensor(12.4239)