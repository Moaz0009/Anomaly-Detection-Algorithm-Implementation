# Anomaly Detection Algorithm Implementation

Welcome to the Anomaly Detection Algorithm repository! This project contains the implementation of an anomaly detection algorithm using Gaussian distribution-based probability estimation. The algorithm is designed to detect outliers in a dataset and determine an optimal threshold for identifying anomalies.

## Table of Contents
- [Overview](#overview)
- [Algorithm Details](#algorithm-details)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Example](#example)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This repository contains the implementation of an anomaly detection algorithm based on the Gaussian distribution of features. The algorithm identifies the optimal threshold (epsilon) for classifying data points as anomalies. The repository also includes a skewed dataset to test the effectiveness of the algorithm.

## Algorithm Details
The algorithm estimates the probability of each feature in the dataset using a Gaussian distribution, then combines these probabilities to compute the overall probability for each data point. Based on a threshold (epsilon), the algorithm labels each data point as normal or anomalous.

### Key Features:
- **Probability Estimation:** Calculates the probability of each data point using the Gaussian distribution.
- **Threshold Selection:** Finds the optimal epsilon value for anomaly detection using a validation set.
- **Skewed Data Handling:** Demonstrates performance on skewed datasets, which are common in anomaly detection problems.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- Matplotlib (for visualization)
- Any other relevant libraries

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Moaz0009/Anomaly-Detection-Algorithm-Implementation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Anomaly-Detection-Algorithm-Implementation
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Example
```python
# Import necessary libraries
import numpy as np

# Assume the algorithm and functions are already implemented

# Generate a skewed dataset with anomalies
np.random.seed(42)
x_train = np.random.normal(loc=0.0, scale=1.0, size=(1000, 2))
x_val = np.concatenate([np.random.normal(loc=0.0, scale=1.0, size=(98, 2)), np.random.normal(loc=5.0, scale=1.0, size=(2, 2))])
y_val = np.concatenate([np.zeros(98), np.ones(2)])

# Fit the model and detect anomalies
fit(x_train)
epsilon, best_F1 = select_epsilon(x_val, y_val)
predictions = predict(x_val, epsilon)
```

## Testing
To test the algorithm on the provided skewed dataset, you can use the example code snippet in the [Usage](#usage) section. The test will demonstrate how well the algorithm detects anomalies in a dataset with a majority of normal data points and a few anomalies.

## Contributing
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request. Please ensure that your code follows the project's coding standards.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions, feel free to reach out.
