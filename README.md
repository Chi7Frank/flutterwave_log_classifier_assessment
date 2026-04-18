# FLUTTERWAVE LOG CLASSIFICATION ASSESSMENT

# Model Approach

I implemented a TF-IDF (Term Frequency-Inverse Document Frequency) vectorization paired with a Logistic Regression classifier. This approach is highly effective for log analysis because log entries typically rely on recurring, specific error tokens (e.g., "502," "OOM," "TLS handshake"). This pipeline provides high interpretability, rapid training times, and sub-second inference latency, which are critical for operational stability.

# Data Preprocessing

To ensure the model learns generalized patterns rather than specific instances, I implemented a custom cleaning function:

- **Normalization:** Converted all text to lowercase to ensure consistency.
- **Noise Reduction:** Used Regular Expressions to strip dynamic identifiers such as IP addresses, transaction IDs, hex codes, and timestamps. This prevents the model from overfitting on unique, non-informative data points.

# Trade-offs & Limitations

- **Contextual Sensitivity:** While TF-IDF is robust, it lacks semantic context. For extremely complex or ambiguous log formats, a transformer-based approach (like BERT) might be superior but would introduce significantly higher latency and resource costs.
- **Data Volume:** With a dataset of ~120 samples, the model is prone to high variance. Future iterations would benefit from an "Active Learning" loop where low-confidence predictions are manually audited and added to the training set.

# Productionization Roadmap

- **Observability:** Monitor prediction confidence scores; any inference below a 0.75 threshold triggers an alert for manual inspection.
- **Drift Detection:** Implement population stability index (PSI) monitoring on incoming logs to detect when the system encounters "out-of-distribution" error types that require model retraining.
- **Scalability:** Deploy the inference module as a microservice using FastAPI inside a Docker container, allowing for horizontal scaling under heavy load.
- **Reliability:** Implement an MLOps lifecycle with model versioning (e.g., using MLflow) to enable seamless rollbacks and A/B testing of updated classification logic.

# Demo Video

- https://drive.google.com/file/d/1PeJaN31OKOK8d1Zfj9DG6dCgVSFc06OY/view?usp=sharing
