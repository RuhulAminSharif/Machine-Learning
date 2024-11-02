<details>
  <summary>What is Machine Learning</summary>

## Machine Learning  
**Machine learning** is a branch of artificial intelligence that enables computers to learn from data and improve their performance on tasks over time without being explicitly programmed. By using algorithms to analyze patterns in data, machines can make predictions or decisions with minimal human intervention.  

**Example**:  
A popular example of machine learning is email spam detection. The model is trained on a dataset containing labeled emails—some marked as spam and others as not spam. By analyzing patterns in the text, subject lines, and sender information, the model learns to identify characteristics of spam emails. Once trained, it can classify new incoming emails as either spam or not spam, helping users keep their inboxes clean and organized.
</details>

<details>
   <summary>AI vs ML vs DL</summary>

## AI vs ML vs DL
| Aspect                     | Artificial Intelligence (AI)                                              | Machine Learning (ML)                                                                                                                                       | Deep Learning (DL)                                                                                                             |
|----------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Definition**             | AI is the overall concept of creating intelligent machines that simulate human intelligence and make decisions.                           | ML is a subset of AI focused on algorithms that learn from data patterns without explicit programming.                                                       | DL is a further subset of ML, utilizing neural networks to learn complex patterns from data, inspired by the human brain.      |
| **Learning Process**       | Involves various approaches to simulate intelligence, often including ML and DL techniques.                              | Trains systems on labeled data to identify patterns and relationships, often through supervised learning.                                                   | Uses deep neural networks with multiple layers to extract complex features, can use both supervised and unsupervised learning. |
| **Focus**                  | Building systems that can think, learn, adapt, and make decisions like humans.                                    | Developing algorithms that allow systems to learn from data and make predictions or decisions based on past experiences.                                    | Specializes in identifying complex patterns, particularly in unstructured data like images, text, and sound.                   |
| **Techniques**             | Rule-based systems, decision trees, expert systems, robotics.                                                   | Supervised, unsupervised, and reinforcement learning algorithms.                                                      | Neural networks, including Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).                           |
| **Data Requirements**      | Varies; can work with rules or limited data but benefits from larger datasets.                                   | Needs large datasets to generalize and make accurate predictions.                                                      | Requires vast datasets to accurately model complex patterns and relationships.                                                |
| **Computational Power**    | Moderate; often runs on CPUs, sometimes with GPU support.                                                      | Higher than traditional AI, can benefit from GPU support for larger datasets.                                          | Very high; relies on GPUs or TPUs for handling large amounts of data and complex computations.                               |
| **Example in Self-Driving Cars** | Combines outputs from ML and DL to make driving decisions, plan routes, control speed, and interact with passengers.               | Identifies objects on the road, predicts other vehicles’ behavior based on historical data, assists with object detection and obstacle avoidance.           | Recognizes objects in images, analyzes sensor data to detect obstacles and anticipate changes in the environment.              |
| **Real-Life Use Case**     | Virtual assistants (Siri, Alexa), facial recognition, recommendation systems.                                  | Spam detection, credit scoring, product recommendations.                                                              | Self-driving cars, medical imaging analysis, natural language processing.                                                     |
| **Overall Role in AI**     | AI is the broadest category, encompassing ML and DL as methods to achieve intelligent systems.                 | ML is a technique within AI to enable systems to improve based on data without explicit programming.                    | DL is a specialized ML approach effective in handling unstructured data and complex relationships.                            |


</details>
<details>
  <summary>Application of Machine Learning</summary>
we will see the application of machine learning in B2B and B2C contexts:

### B2B Applications of Machine Learning
1. **Supply Chain Management**: Machine learning optimizes inventory levels, predicts demand, and enhances logistics efficiency. It helps businesses like Walmart streamline operations and reduce costs.
2. **Customer Relationship Management (CRM)**: ML algorithms help in personalizing marketing campaigns, predicting customer churn, and improving customer satisfaction. Salesforce, for example, uses ML to provide insights and automate tasks.
3. **Cybersecurity**: ML detects anomalies, identifies threats, and enhances data security. IBM and other cybersecurity firms use ML to provide advanced threat protection for businesses.
4. **Financial Services**: In B2B banking, ML assists in fraud detection, credit risk assessment, and investment forecasting. Companies like Bloomberg use ML for predictive analytics.
5. **Manufacturing**: ML is used in predictive maintenance, quality control, and process optimization. Siemens leverages ML to monitor equipment health and optimize production lines.

### B2C Applications of Machine Learning
1. **Retail**: Machine learning powers product recommendations, optimizes pricing, and personalizes the shopping experience for users. For example: Amazon
2. **Banking and Finance**: ML is employed for fraud detection, credit scoring, and personalized financial advice, making transactions more secure and services more tailored.
3. **Transport**: Machine learning optimizes route planning, predicts demand, and improves ride-sharing efficiency to provide better customer experiences. For example: OLA
4. **Manufacturing**: Tesla uses ML for autonomous driving, predictive maintenance, and improving vehicle safety by analyzing sensor data and driver behavior. For example: Tesla
5. **Consumer Internet**: ML helps in content recommendation, spam detection, and trend analysis, ensuring a more personalized and secure experience for users. For example: Twitter

</details>

<details>
  <summary>Machine Learning Development Life Cycle (MLDLC)</summary>

MLDLC is a framework for developing machine learning models in a structured and systematic way. It includes several steps, from problem definition to deployment, that are focused to build robust, accurate, and scalable machine learning models.

Here’s a step-by-step guide to solving a machine learning problem using the Machine Learning Development Cycle (MLDC), illustrated with an example of predicting house prices:

### 1. **Frame the Problem**
   - **Objective**: Define the problem, goals, and performance metrics.
   - **Example**: Suppose we want to build a model that predicts house prices based on features like bedrooms, bathrooms, and location. The goal is to create a model that accurately predicts the price, using metrics like Mean Squared Error (MSE) to evaluate performance.

### 2. **Data Collection**
   - **Objective**: Collect relevant data from various sources.
   - **Example**: For predicting house prices, data can be sourced from real estate websites, public datasets, or web scraping tools to obtain information on home features and sale prices.  
  
(Data can be in csv format, or can be collected using API, web scraping, database to data ware house via ETL.)

### 3. **Data Preprocessing**
   - **Objective**: Clean and prepare data for modeling.
   - **Example**: Remove duplicate values, remove rows with missing values, encode categorical features like location using encoding, and scale numerical features like square footage and lot size to standardize them, remove outliers.

### 4. **Exploratory Data Analysis (EDA)**
   - **Objective**: Understand the data distribution and relationships among features.
   - **Example**: Visualize relationships using scatter plots (e.g., between square footage and price) and histograms to see the distribution of features and identify patterns that influence house prices.
  
(visualization, univariate, bi-variate, multivariate, outlier detection, imbalance -> balance)

### 5. **Feature Engineering and Selection**
   - **Objective**: Create and select features that improve model performance.
   - **Example**: Create new features like “house age” or “bathroom-to-bedroom ratio.” Use correlation analysis or feature importance scores to select the most impactful features for predicting prices.

### 6. **Model Training, Evaluation, and Selection**
   - **Objective**: Train models, evaluate their performance, and select the best one.
   - **Example**: Train models like Linear Regression and Decision Tree Regression. Evaluate them using metrics such as MSE or R-squared, and select the model with the best performance based on these metrics.

### 7. **Model Deployment**
   - **Objective**: Deploy the model in a production environment.
   - **Example**: Deploy the house price prediction model on a website or app where users can input house features and receive a predicted price.

### 8. **Testing and Optimization**
   - **Objective**: Continuously test and optimize the model in production.
   - **Example**: Regularly test the model on new data to ensure accuracy. If performance drops, retrain or update the model with new data or features to improve accuracy.

By following these steps, we can systematically build, deploy, and maintain a machine learning model that solves real-world problems effectively.
</details>