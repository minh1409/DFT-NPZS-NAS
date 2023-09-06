import re
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def learning_curve(epochs, models):
    """
    Visualize the learning curves of different models over a series of epochs.

    Parameters:
    epochs (list): List of epoch values (x-axis) representing the training iterations.
    models (dict): A dictionary where keys are model names or benchmarks, and values
                   are lists of correlation values corresponding to each epoch for that model.

    This function generates a line plot for each model's learning curve, showing how the
    correlation values change over the specified epochs. It allows for easy comparison
    of different models' performance during training.

    The function sets the x-axis as 'Epoch', the y-axis as 'Correlation', and provides
    appropriate labels and ticks for better readability. It also includes a legend to
    distinguish between the different models being compared.

    Example Usage:
    epochs = [16, 80, 176, 352, 528, 720, 896, 1072, 1248 ,1440]
    models = {
        'DARTS': [0.2, 0.4, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.9, 0.9],
        'NASNet': [0.1, 0.3, 0.5, 0.1, 0.3, 0.1, 0.8, 0.7, 0.8, 0.9],
        'ENAS': [0.3, 0.5, 0.6, 0.5, 0.5, 0.5, 0.7, 0.7, 0.85, 0.85, 0.85]
    }
    learning_curve(epochs, models)
    plt.show()
    """
    # Importantly, this function assumes that 'matplotlib.pyplot' (plt) has been imported
    # elsewhere in your code to create the visualization.
    # Uncomment the 'fig' line and the import if necessary, and modify the size as needed.
    
    # fig = plt.figure(figsize=(10, 7))
    
    # Iterate over each benchmark or model in the dictionary.
    for benchmark_train in models:
        # Plot the model's learning curve as a line with markers.
        plt.plot(epochs, models[benchmark_train], '-o', label=benchmark_train)

    # Set the x and y-axis labels with appropriate font sizes.
    plt.xlabel('Epoch', fontsize="20")
    plt.ylabel('Correlation', fontsize="20")
    
    # Define specific tick values for the y-axis and use the provided epochs for x-axis ticks.
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(epochs)

    # Set the title of the plot.
    plt.title('Learning Curve', fontsize="20")
    
    # Add a legend to distinguish between different models.
    plt.legend(fontsize="16")
    
    # Display the plot.
    plt.show()

