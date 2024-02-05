import matplotlib.pyplot as plt

from models.data_generation import generate_synthetic_data

x, y_with_noise, y_bernoulli = generate_synthetic_data()


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y_with_noise, s=2, alpha=0.7)
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Synthetic Data with Bimodal Distribution")
plt.show()
plt.close()