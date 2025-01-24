## Label Encoding
The `LabelEncoder` class in scikit-learn is a simple and effective tool for converting categorical data into numerical labels. 


### **What is LabelEncoder?**
The `LabelEncoder` is a preprocessing class in `sklearn.preprocessing` that encodes categorical labels (strings or other types) into numerical values. It is widely used to prepare target variables for machine learning models, particularly classification tasks.

### **Key Features**
1. **Encodes Categories to Integers**:
   - Maps each unique class to a unique integer value (e.g., "cat" → 0, "dog" → 1).
   - Ensures consistent mapping during encoding and decoding.
   
2. **Bidirectional Mapping**:
   - Supports both encoding (original → integer) and decoding (integer → original).

3. **Handles Single Feature**:
   - Operates on a single array of categorical data (e.g., a target variable).
   - Does not directly support multi-column or multi-dimensional arrays.



### **When to Use LabelEncoder**
- **Target Variable Encoding**: Used to encode categorical target variables (e.g., class labels) into numerical values for machine learning algorithms.
- **Avoid for Features**: Not recommended for encoding input features, as it introduces ordinality (e.g., 0 < 1 < 2), which may mislead the model.



### **How LabelEncoder Works**
1. **Fit the Data**:
   - Learn the unique categories from the data.
   - Create a mapping between original categories and numerical labels.
   
2. **Transform the Data**:
   - Convert original categories to numerical labels based on the mapping.

3. **Inverse Transform**:
   - Convert numerical labels back to the original categories.

---

## **Methods in LabelEncoder**

### 1. **`fit(y)`**
   - Fits the encoder by learning the unique classes from the input array `y`.
   - Parameters:
     - `y`: Array-like input containing the categorical data.
   - Stores the classes in the `classes_` attribute.
   - Example:
     ```python
     from sklearn.preprocessing import LabelEncoder
     encoder = LabelEncoder()
     encoder.fit(["red", "blue", "green"])
     print(encoder.classes_)  # Output: ['blue' 'green' 'red']
     ```

### 2. **`transform(y)`**
   - Transforms the input `y` into its corresponding numerical labels.
   - Parameters:
     - `y`: Array-like input to be transformed (must contain values seen during `fit()`).
   - Returns an array of encoded integers.
   - Example:
     ```python
     encoded = encoder.transform(["red", "green", "blue"])
     print(encoded)  # Output: [2, 1, 0]
     ```

### 3. **`fit_transform(y)`**
   - Combines `fit` and `transform` in one step.
   - Parameters:
     - `y`: Array-like input containing categorical data.
   - Example:
     ```python
     encoded = encoder.fit_transform(["red", "blue", "green", "blue"])
     print(encoded)  # Output: [2, 0, 1, 0]
     ```

### 4. **`inverse_transform(y)`**
   - Converts encoded integers back into their original categories.
   - Parameters:
     - `y`: Array-like input containing numerical labels to decode.
   - Example:
     ```python
     original = encoder.inverse_transform([2, 1, 0])
     print(original)  # Output: ['red' 'green' 'blue']
     ```

### 5. **`classes_` Attribute**
   - Stores the unique classes found during `fit()`.
   - Example:
     ```python
     print(encoder.classes_)  # Output: ['blue' 'green' 'red']
     ```



### **Limitations**
1. **Implicit Ordinality**:
   - Introduces an ordinal relationship between classes (e.g., `0 < 1 < 2`), which may mislead the model if the data is nominal.

2. **Not Suitable for Features**:
   - Use `OneHotEncoder` for encoding feature variables to avoid ordinal relationships.

3. **Error on Unseen Labels**:
   - Throws an error if unseen labels are passed during `transform()`.



### **Tips for Using LabelEncoder**
1. **Use for Target Variables**:
   - Ideal for classification tasks to encode labels.
   
2. **Handle Unseen Categories**:
   - Use `.classes_` to ensure all potential categories are included during `fit`.

3. **Avoid for Features**:
   - Use `OneHotEncoder` or `pd.get_dummies()` for input features.
