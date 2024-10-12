In this project, we implemented and evaluated three different machine learning models to develop an anime recommendation system: K-Nearest Neighbors (KNN), Funk Singular Value Decomposition (SVD), and Factorization Machines (FM). The performance of these models was assessed by comparing their Root Mean Square Error (RMSE) values on the training, validation, and test datasets.

### Key Findings:

### 1.	K-Nearest Neighbors (KNN):
	•	RMSE on training data: 1.18
	•	RMSE on validation data: 1.17
	•	RMSE on test data: 1.44
•	KNN demonstrated strong performance with a shorter execution time compared to the other models. The final model, with 19 nearest neighbors, used Mean Squared Error as the similarity measure.

### 2.	Funk SVD:
	•	RMSE on training data: 1.24
	•	RMSE on validation data: 1.23
	•	RMSE on test data: 1.44
 •	While Funk SVD achieved similar accuracy to KNN, it required a significantly longer training time due to the iterative process of matrix factorization and gradient descent optimization.

### 3.	Factorization Machines (FM):
 •	Due to technical challenges in setting up cross-validation for the FM model, the results were not fully generated, and the model could not be properly compared with KNN and Funk SVD.

### Conclusion:

Both KNN and Funk SVD produced comparable RMSE values, but KNN was selected as the best-performing model due to its lower training time and ability to deliver predictions more efficiently. For future improvements, we suggest further optimizing the FM model and exploring additional hybrid techniques to combine the strengths of both KNN and SVD for more robust recommendations.
