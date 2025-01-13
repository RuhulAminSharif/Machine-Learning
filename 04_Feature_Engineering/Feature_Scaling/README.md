### Feature Scaling

#### What is Feature Scaling?
Feature scaling is a technique used to normalize the range of independent variables or features of data. It transforms the data to fit within a specific scale, such as 0 to 1 or -1 to 1. This is an essential preprocessing step in many machine learning algorithms to ensure that each feature contributes equally to the model.

#### Why Do We Need Feature Scaling?
1. **Improves Model Performance**: Many machine learning algorithms perform better when the features are on a similar scale. Algorithms like gradient descent converge faster when the features are scaled.
2. **Prevents Dominance**: Features with larger ranges can dominate the model's learning process, leading to biased results. Scaling ensures that no single feature disproportionately affects the model.
3. **Facilitates Distance-Based Algorithms**: Algorithms like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM) rely on distance metrics. Feature scaling ensures that all features contribute equally to the distance calculations.
4. **Enhances Interpretability**: Scaled features can be more easily interpreted, especially in regression models.

#### Types of Feature Scaling

1. **Standardization (Z-score Normalization)**
   - **Definition**: Standardization transforms the data to have a **mean of 0 and a standard deviation of 1**.  
   It is done using the formula:  
    <p align="center">
        <code>Z = (X - μ) / σ</code>
    </p>
    
    where `x` is the feature value, `μ` is the mean of the feature, and `σ` is the standard deviation of the feature.
   - **When to Use**: Standardization is useful when the data follows a Gaussian (normal) distribution. It is commonly used in algorithms that assume normally distributed data, such as linear regression, logistic regression, and linear discriminant analysis.
   - **Types of Standardization**:
    - **Z-score Standardization**: This is the most common form of standardization described above.
    - **Mean Normalization**: Similar to Z-score but normalizes to a range around the mean:  
    <p align="center">
        <code>X<sub>norm</sub> = (X - μ) / ( X<sub>max</sub> - X<sub>min</sub> )</code>
    </p>

     - **Unit Vector**: Scales the feature vector to have a length of 1:
    
    <p align="center">
        <code>X<sub>unit</sub> = X / | X |</code>
    </p>

2. **Normalization (Min-Max Scaling)**
   - **Definition**: Normalization rescales the data to fit within a specific range, typically [0, 1]. It is done using the formula:  
     <p align="center">
     <code>X' = (X - X<sub>min</sub>) / (X<sub>max</sub> - X<sub>min</sub>)</code>
     </p>
     
    where X is the original feature value, X<sub>min</sub> and X<sub>max</sub> are the minimum and maximum values of the feature, respectively.
   - **When to Use**: Normalization is useful when the data does not follow a Gaussian distribution or when you want to bound the feature values within a specific range. It is commonly used in algorithms like neural networks and K-Means clustering.
   - **Types of Normalization**:
     - **Min-Max Normalization**: The most common form, which scales data to a range of [0, 1] or [-1, 1].
     - **Max-Abs Scaling**: Scales each feature by its maximum absolute value:  
    
    <p align="center">
        <code>X<sub>norm</sub> = X / abs( X<sub>max</sub> ) </code>
    </p>
    
    - **Robust Scaling**: Uses the median and interquartile range (IQR) to scale features, which is robust to outliers:  

    <p align="center">
        <code>X<sub>norm</sub> = ( X - median(X) ) / IQR( X ) </code>
    </p>