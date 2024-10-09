# **Bank Loan Granting Prediction using Deep Learning**

## Overview
This project develops a deep learning model using **TensorFlow Keras** to predict bank loan granting decisions based on the **Bank_Loan_Granting.csv** dataset. The goal is to classify loan applications into multiple categories using a baseline architecture and an optimized model.

## Project Structure
1. **Baseline Architecture**: 
   A model built from scratch following the structure:
   - Input Layer: `n` nodes (equal to the number of input features).
   - Two Hidden Layers: Each with `2 × n` nodes, using **ReLU** activation.
   - Output Layer: `num_class` nodes (equal to the number of output classes), using **softmax** for multi-class classification.
   
   This model was implemented as a baseline to assess the initial performance of the data.

2. **Optimized Architecture**: 
   Based on the evaluation of the baseline model, several optimizations were made to improve accuracy:
   - **Feature Scaling**: Applied **Min-Max Scaling** to normalize input features.
   - **Class Weights**: Used to address the **class imbalance** present in the dataset.
   - **Hyperparameter Tuning**: Adjusted parameters like learning rate, batch size, and optimizer type for enhanced performance.
   
   The optimizations significantly improved the accuracy and resolved issues related to the class imbalance.

## Key Challenges
- **Class Imbalance**: The dataset had uneven class distribution, leading to biased predictions.
- **Normalization**: The features required scaling to enhance model convergence.

## Technologies Used
- **Python**: Core language for data processing and model building.
- **TensorFlow Keras**: For constructing and training the deep learning models.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: Used for additional preprocessing steps like feature scaling.
  
## Results
- **Baseline Model**: Initial model with moderate accuracy.
- **Optimized Model**: Improved accuracy by using **class weights** and **scaling techniques** to handle class imbalance.

## Future Work
- Further optimize the architecture with advanced techniques such as **dropout** or **batch normalization**.
- Explore other deep learning frameworks or ensemble methods to boost accuracy.

## Acknowledgments
Thanks to the open-source community for providing powerful tools like **TensorFlow Keras**, **Pandas**, and **Scikit-learn** for enabling this project.
