# 23CSE301 Machine Learning: Capstone Project

Guidelines, algorithm list and evaluation rubrics.
B.Tech. Computer Science and Engineering, III Year. Academic year 2026-27.

## What the project asks for

Each team builds one complete machine learning pipeline and runs it across three problem types: regression, classification and clustering. Everything from loading the data through to the final plots is part of the deliverable. That means EDA, cleaning, feature engineering, training and comparing models, tuning them, and visualising what you found. The dataset and the problem statement come from the instructor, one dataset per track per team.

Teams are three people. There are two reviews worth 25 marks each, so 50 in total. Work in Python 3 with scikit-learn, Pandas, NumPy, Matplotlib and Seaborn. You submit Jupyter notebooks plus a GitHub repository.

## When the reviews happen

Review 1 falls just before the mid-semester exams and covers the whole regression track plus Part A of classification. Review 2 is near the end of the semester and covers Part B of classification plus all of clustering.

Both are in-person presentations to the instructor. Everyone on the team should be there, and anyone can be asked about any part of the work, not just their own section.

## Algorithms

### Regression (Review 1)

Train all ten on the same preprocessed data and evaluate them on the same held-out test set. Comparing models that saw different splits tells you nothing.

| # | Algorithm | Notes |
|---|---|---|
| 1 | Linear Regression | Baseline; interpret coefficients |
| 2 | Ridge Regression | L2 regularisation; tune `alpha` |
| 3 | Lasso Regression | L1 regularisation; observe feature sparsity |
| 4 | ElasticNet Regression | Combined L1+L2; tune `l1_ratio` |
| 5 | Polynomial Regression | Apply `PolynomialFeatures` then Linear Regression; compare degrees |
| 6 | Decision Tree Regressor | Tune `max_depth`; show feature importance |
| 7 | Random Forest Regressor | Ensemble baseline; tune `n_estimators` |
| 8 | Gradient Boosting Regressor | sklearn GBM or XGBoost; tune learning rate |
| 9 | Support Vector Regressor (SVR) | Scale features first; tune `C` and kernel |
| 10 | K-Nearest Neighbors Regressor | Tune `k`; discuss impact of scaling |

Report R² score, RMSE and MAE for every model. All three are compulsory. On top of that, run 5-fold cross-validated R² for your two best performers.

### Classification (Part A in Review 1, Part B in Review 2)

| # | Part | Algorithm | Notes |
|---|---|---|---|
| 1 | A | Logistic Regression | Baseline classifier; interpret coefficients/odds |
| 2 | A | K-Nearest Neighbors | Tune `k`; discuss distance metrics |
| 3 | A | Naive Bayes (Gaussian) | Discuss conditional independence assumption |
| 4 | A | Decision Tree Classifier | Tune `max_depth`; visualise the tree |
| 5 | A | Support Vector Machine (SVC) | Tune `C` and kernel; scale features |
| 6 | B | Random Forest Classifier | Feature importance; tune `n_estimators` |
| 7 | B | AdaBoost Classifier | Tune `n_estimators` and learning rate |
| 8 | B | Gradient Boosting Classifier | sklearn GBM or XGBoost/LightGBM |
| 9 | B | Bagging Classifier | Use Decision Tree as base estimator |
| 10 | B | MLP Classifier (Neural Network) | Tune `hidden_layer_sizes` and activation |

Required metrics: accuracy, precision, recall, weighted F1, a confusion matrix, and ROC-AUC. If the problem is multi-class, use One-vs-Rest for the ROC-AUC. By Review 2 all ten algorithms have to appear in one consolidated comparison table.

### Clustering (Review 2)

Clustering has no labels to learn from, which changes how you work. If the dataset has ground-truth labels, use them afterwards to interpret or sanity-check the clusters. Never during fitting.

| # | Algorithm | Notes |
|---|---|---|
| 1 | K-Means Clustering | Plot Elbow curve (inertia vs. k) to choose k |
| 2 | Agglomerative Hierarchical Clustering | Plot Dendrogram; compare linkage strategies |

Report Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index for both algorithms. A PCA plot reduced to two components is required; t-SNE is optional but worth doing.

## Deliverables

| # | Deliverable | Review 1 | Review 2 |
|---|---|---|---|
| D1 | Jupyter Notebook(s), clean, fully run, outputs visible | yes | yes |
| D2 | GitHub repository with meaningful commit history | yes | yes |
| D3 | Comparative results table (algorithms vs. metrics) | yes | yes |
| D4 | All required visualisations (see rubric) | yes | yes |
| D5 | README file (dataset description, instructions to run) | no | yes |
| D6 | Consolidated 10-algorithm classification comparison | no | yes |
| D7 | Cluster visualisations (Elbow, Dendrogram, PCA plot) | no | yes |
| D8 | GUI / deployed application | no | optional bonus |


## General guidelines

### Code and notebooks

Use one notebook per track, named something like `regression.ipynb`, `classification.ipynb` and `clustering.ipynb`. A single notebook is fine too as long as the sections are clearly labelled.

Run every cell top to bottom before you submit and leave the outputs visible. Notebooks with execution errors in them will cost you marks. Explain each major code block in a Markdown cell rather than dumping a wall of code and hoping it speaks for itself. Set `random_state=42` anywhere it applies so your numbers are reproducible.

One rule worth repeating: fit scalers and encoders on the training set only, then transform both train and test with them. Fitting on the full dataset leaks test information into training and will be marked down.

### Model evaluation

Keep the same train/test split, usually 80:20, across every algorithm in a track. Otherwise the comparison is meaningless. Cross-validate at least your top two models per track. Put results in one summary table, either a Pandas DataFrame displayed in the notebook or a formatted Markdown table, rather than scattering print statements through the file.

### Visualisations

Every plot needs a title, labelled axes and a legend where one makes sense. Call `plt.tight_layout()` or save with `bbox_inches='tight'` so labels don't get clipped. Colourblind-friendly palettes like `tab10`, `Set2` or Seaborn's `colorblind` are strongly recommended.






