from matplotlib import pyplot as plt


import matplotlib.pyplot as plt

# Categories and their accuracies
categories = ['Normal Dropout', 'MCMC Dropout', 'Normal MLP', 'Fixed Dropout Masks']
accuracies = [0.9156, 0.924, 0.9197, 0.912]  # Example accuracy percentages

# Create the bar plot
plt.figure(figsize=(8, 5))
plt.bar(categories, accuracies)

# Labels and title
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Models with deeper network')
plt.ylim(0.75, 1)

# Show values on top of bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc}%", ha='center')

# Display the plot
plt.savefig('deep_network.png')