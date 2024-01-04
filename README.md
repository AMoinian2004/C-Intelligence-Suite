# C++-Intelligence-Suite

Here's a demonstration of my library: [Cpp-Intelligence-Suite Demo](https://www.youtube.com/watch?v=_ABj1-VvLdw&feature=youtu.be)

## To Download Using GitHub
1. Click the "Download ZIP" button under the green "Code" button.
2. Unzip the file.
3. Move all contents inside the folder into the folder of the C++ file you want to use it in.
4. Write `#include "MLLibrary.h"` at the top of the C++ file you want to use it in.

**NOTE**: The `main.cpp` file includes test cases. It is advised for you to delete the `main.cpp` file, unless you don't have a `main.cpp` file in your project already and want to use the test cases.

## To Download Using Docker
To use the library in a Docker environment, pull the Docker image and integrate it into your project: 
```Dockerfile
docker pull amoinian/mllibrary:latest
```

You can run a container from this image and navigate to `/usr/src/app` to find the `MLLibrary.h`. Additionally, you can use the image as a base in your Dockerfile:

```Dockerfile
FROM amoinian/mllibrary:latest
COPY . /usr/src/myapp
WORKDIR /usr/src/myapp
RUN g++ -o myapp main.cpp
CMD ["./myapp"]
```

## Features
### Linear Regression
- Fitting
- Predicting

### Logistic Regression
- Binary Classification
- Model Training with Configurable Learning Rate and Iterations

### Decision Tree
- Classification
- Configurable Maximum Tree Depth

### K-Means Clustering
- Unsupervised Clustering
- Configurable Number of Clusters and Iterations

### Data Splitting
- Train-Test Data Separation
- Configurable Split Ratio

### Neural Network
- Basic Neural Network for Simple Tasks (e.g., XOR problem)
- Configurable Input, Hidden, and Output Layer Sizes
- Training Functionality

### Data Normalization & Standardization
- Min-Max Scaling (Normalization)
- Z-Score Normalization (Standardization)

### Monte Carlo Simulation
- Probability and Statistical Analysis
- Configurable Number of Samples and Threads

### Testing Loss Functions
- Mean Squared Error (MSE) for Regression
- Cross-Entropy Loss for Classification
