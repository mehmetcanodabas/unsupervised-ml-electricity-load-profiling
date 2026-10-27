# unsupervised-ml-electricity-load-profiling

Unsupervised Electricity Consumption Pattern Discovery Using Smart Meter Data, Feature Engineering, and K-Means Clustering

This project identifies customer electricity consumption patterns using unsupervised machine learning and half-hourly smart meter measurements from the London Low Carbon London (LCL) dataset.

Electricity consumption records from 2013 were processed to extract nine customer-level behavioral features: six four-hour time-of-day consumption shares, a weekend-to-weekday consumption ratio, a winter-to-summer consumption ratio, and daily consumption variability.

Feature engineering and exploratory analysis were conducted to investigate customer behavior, extreme consumption patterns, feature correlations, and clustering alternatives. Log transformations, StandardScaler normalization, and feature-group weighting were applied before clustering.

The model was trained using 3,208 development customers and evaluated on 357 held-out test customers. A reusable inference pipeline was implemented to assign customers with precomputed behavioral features to the learned clusters without retraining the model.

### Key Results

- Processed 2013 half-hourly electricity consumption data from the London Low Carbon London (LCL) dataset.
- Selected 3,569 eligible customers and prepared **3,208 development** and **357 held-out test customers**.
- Engineered **nine behavioral features** describing time-of-day consumption, weekly and seasonal patterns, and daily variability.
- Evaluated alternative clustering configurations and selected **K-Means (K=3)** based on cluster quality and interpretability.
- Achieved silhouette scores of **0.2531 (development)** and **0.2414 (held-out test)**.
- Identified three behavioral segments: **Weekend-Intensive (635)**, **Balanced / Regular (2,176)**, and **Winter-Dominant (397)**.
- Evaluated clustering stability using repeated subsampling, obtaining a mean **Adjusted Rand Index (ARI) of 0.8121**.
- Saved the trained model and preprocessing pipeline, implemented reusable customer profiling, and validated inference on all **357 held-out customers** with zero cluster assignment mismatches.

### Identified Consumption Segments

**Cluster 0 — Weekend-Intensive / Evening-Oriented**

Customers exhibiting relatively higher weekend electricity consumption and greater consumption shares during evening hours.

**Cluster 1 — Balanced / Regular**

Customers exhibiting comparatively balanced weekly consumption patterns, lower daily variability, and moderate seasonal differences.

**Cluster 2 — Winter-Dominant / High-Variability**

Customers exhibiting substantially higher winter consumption relative to summer, together with greater day-to-day consumption variability.

### Result Snapshots

[Electricity Consumption Behavioral Segments](docs/results/cluster_profiles.png): Visualization of the three K-Means clusters using median time-of-day consumption shares and weekend-to-weekday and winter-to-summer consumption ratios. The figure includes the development customer counts for each cluster.

**Keywords:** Unsupervised Machine Learning, Electricity Load Profiling, Smart Meter Analytics, Energy Consumption Analysis, Customer Segmentation, K-Means Clustering, Feature Engineering, Time-of-Day Consumption, Seasonal Consumption, Energy Analytics, Scikit-learn, Python.

**Data Source:** Low Carbon London (LCL) Smart Meter Electricity Consumption Dataset, half-hourly electricity measurements from 2013.
