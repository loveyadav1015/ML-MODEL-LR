# 🏡 Linear Regression From Scratch: California Housing

An end-to-end Machine Learning project demonstrating the core mathematics of Linear Regression. Instead of relying on pre-packaged libraries like `scikit-learn` to do the heavy lifting, I built the entire learning algorithm—including the Cost Function, Gradient Descent, and Feature Scaling—completely from scratch using Python and it's usefull librarys like pandas , numpy and matplotlib.

## 🧠 Core Features & Algorithms Built

* **Vectorized Gradient Descent:** Implemented the parameter update rules using pure NumPy dot products for highly optimized matrix multiplication, avoiding computationally expensive `for` loops .
* **Cost Function (Mean Squared Error):** Programmed the mathematical logic to calculate the exact error distance between the model's hyper-plane and the actual dataset.
* **Custom Data Preprocessing:** Built a Z-Score Normalization (Standardization) pipeline from scratch to dynamically scale features, preventing memory overflows and ensuring the algorithm converged rapidly.

## 📊 The Dataset

This model is trained on the **California Housing Dataset**. It predicts the median house value (`y`) based on 9 distinct continuous features (`X`):
* Longitude & Latitude
* Housing Median Age
* Total Rooms & Total Bedrooms
* Population & Households
* Median Income & Population_per_household

## 🚀 Results & Visualizations

The model successfully navigated the error gradient to find the optimal weights and biase. 

* **Rapid Convergence:** The custom learning rate and scaled data allowed the model to shave billions of dollars off the initial error within the first 150 iterations.
* **Evaluation:** Evaluated against a 20% holdout test set to ensure the model generalizes to unseen data, graded using R-squared ($R^2$) and Root Mean Squared Error (RMSE).
* **Visual Proof:** Analyzed the distribution of the data before and after cleaning/scaling (see `PictureDataSet/` for visual documentation).

## 🛠️ Tech Stack

* **Python 3**
* **NumPy:** For core matrix mathematics and vectorized equations.
* **Pandas:** For initial DataFrame manipulation and feature extraction.
* **Matplotlib:** For visualizing the learning curve and plotting actual vs. predicted pricing scatters.
* **scikit-learn:** Used *only* for the `train_test_split` partitioning and final $R^2$ accuracy verification.

## 💡 How to Run
1. Clone the repository to your local machine.
2. Ensure you have `numpy`, `pandas`, and `matplotlib` installed.
3. Run the `model.ipynb` Jupyter Notebook from top to bottom to watch the algorithm scale the data, train the weights, and plot the final accuracy graphs.

##  Future Aspects

* Going to build more ML models.
* Integrate Model with web applications.
