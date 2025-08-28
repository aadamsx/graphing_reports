# Machine Learning Study Guide — Concepts, Terms, Intuition, and Minimal Math

> Purpose: a copy‑friendly, NotebookLM-friendly reference. Every entry says **what it is**, **why it matters**, and gives a **quick example or tip**. Core formulas appear in plain text so they paste cleanly.

---

## How to use this guide
1) Skim section headers to place new ideas in a mental map.  
2) For each term: read *what it is* → *why it matters* → *example/tip*.  
3) When unsure about a symbol, search within this file; many recur across sections.  
4) Revisit “Probability quickstart” and “Time‑series primer” often; they unlock most ML.

---

## Probability quickstart (ties everything together)

- **Random variable**: a quantity whose value is uncertain.
- **Distribution**: the pattern of values a random variable takes (Normal, Bernoulli, Poisson, etc.).
- **Likelihood**: L(θ) = p(data | θ). Training often **maximizes** log‑likelihood or **minimizes** negative log‑likelihood (NLL).
- **Prior / Posterior**: prior p(θ), posterior p(θ|D) ∝ p(D|θ)p(θ). *MAP* ≈ MLE + regularization from the prior.
- **Predictive distribution**: p(ŷ | x, D). Decisions use expected value and cost. Example: choose threshold τ that maximizes expected profit.
- **Uncertainty**: *aleatoric* = irreducible noise; *epistemic* = model ignorance (reduced with more data, ensembling, Bayes).

**Key formulas (plain text):**  
- Standardization: z = (x − mean) / std.  
- Logistic regression probability: p(y=1|x) = 1 / (1 + exp(−w·x − b)).  
- Cross‑entropy (binary): CE = −[y log p + (1−y) log(1−p)]  (this is NLL for Bernoulli).  
- Mean squared error: MSE = average((y − ŷ)^2).  
- Bayes rule: p(θ|D) ∝ p(D|θ) p(θ).

---

## Core concepts

**Dataset** — Collection of samples (rows). Keep a data dictionary that explains each column, units, and allowed ranges.  
**Sample / instance** — One row/observation (a user, a month, an image).  
**Feature** — Input variable used by a model. Numeric, categorical, text, pixels. *Tip:* test feature pipelines; bugs here cause leakage.  
**Label / target** — The variable to predict (price, class, probability).  
**Supervised learning** — Learn mapping x → y from labeled pairs. *Example:* predict churn from user history.  
**Unsupervised learning** — Find structure without labels (clustering, dimensionality reduction).  
**Self‑supervised learning** — Create labels from data itself (mask words, next token). Foundation for LLMs and many vision models.  
**Semi‑supervised** — Combine a few labels with many unlabeled examples via consistency/regularization.  
**Reinforcement learning (RL)** — Learn a policy to maximize reward via trial and error. Concepts: state, action, reward, value, policy.  
**Online vs batch learning** — Update continuously vs retrain on snapshots. Choose by drift and latency requirements.  
**Transfer learning** — Start from a pretrained model; fine‑tune for your task. Saves data and compute.  
**Multitask learning** — One backbone predicts several targets; often improves generalization.  
**Few‑shot / zero‑shot** — Perform with little or no task‑specific data via priors or prompts.

---

## Data preparation

**Train/validation/test split** — Fit/tune/final check. For time series, split *chronologically*.  
**Cross‑validation (CV)** — Average over K folds. For sequences, use rolling/blocked CV rather than random shuffles.  
**Stratification** — Keep class ratios during splits to avoid skewed metrics.  
**Leakage** — Future/test info in training (e.g., using post‑event features). Guard with clean pipelines and time‑aware splits.  
**Standardization** — z = (x − mean) / std. Needed by many linear/NN models. Fit transforms on train only.  
**Normalization** — Scale to [0,1] or unit‑norm. Helps distance‑based methods and images.  
**One‑hot encoding** — Binary indicator columns for categories. Safe but wide.  
**Embedding** — Learned dense vectors for categories/tokens; compact and expressive.  
**Imputation** — Fill missing values (mean/median/model). Keep a missing‑indicator flag.  
**Data augmentation** — Plausible perturbations (flips, noise, crops); acts as regularizer.  
**Class imbalance** — Rare positives. Use class weights, focal loss, thresholding via PR‑AUC, or SMOTE.  
**Dimensionality reduction** —  
- PCA: linear projection to top variance directions (fast baseline).  
- UMAP / t‑SNE: nonlinear structure, great for visualization.  
- Feature selection: filter (mutual info), wrapper, or embedded (L1, tree importances).

---

## Models and algorithms

**Linear regression** — y = Xβ + ε. Fast baseline; interpretable coefficients. Watch multicollinearity.  
**Logistic regression** — Linear classifier with probabilistic outputs via sigmoid/softmax; well‑calibrated with regularization.  
**Regularization** — L1 (sparse), L2 (shrinkage), Elastic Net (mix). Reduces overfitting, improves generalization.  
**SVM** — Margin maximization; kernels add nonlinearity. Strong for small/medium tabular data.  
**Decision tree** — If‑else splits; interpretable; overfits alone.  
**Random forest** — Many bagged trees; robust tabular baseline; out‑of‑bag validation.  
**Gradient boosting** — Additive trees fit to residuals (XGBoost, LightGBM, CatBoost). Often SOTA on tabular.  
**k‑NN** — Predict from neighbors; needs scaling; costly at inference.  
**Naive Bayes** — Simple probabilistic baseline for text and counts.  
**k‑means** — Centroid clustering (Euclidean). Choose k via elbow or silhouette.  
**DBSCAN** — Density clustering; finds noise/outliers; no k needed.  
**GMM** — Soft clustering with Gaussian mixtures via EM algorithm.  
**HMM** — Markov chain with hidden states; classic for speech/POS tagging.  
**VAR / VECM** — Multivariate time‑series models (multiple targets, cointegration).

---

## Neural networks

**MLP (feed‑forward)** — Dense layers with nonlinearity (ReLU/GELU). Good for tabular with care.  
**CNN** — Convolutions exploit locality for images/audio/time series.  
**RNN / LSTM / GRU** — Sequence models with recurrence; largely superseded by attention for long contexts.  
**Attention** — Weighted mixing of tokens by similarity; enables long‑range dependencies.  
**Transformer** — Stacks of attention + MLP + residual + normalization; backbone of LLMs/ViT.  
**Encoder / decoder** — Compress to latent then reconstruct/generate (BERT=encoder, GPT=decoder).  
**Dropout** — Randomly zero units during training; regularization.  
**Batch / Layer norm** — Normalize activations to stabilize and speed training.  
**Common losses** — MSE/MAE/Huber (regression), cross‑entropy/BCE (classification). Label smoothing improves calibration.

---

## Optimization

**Gradient descent / SGD** — Follow negative gradient of loss.  
**Momentum / Nesterov** — Add velocity to accelerate descent and reduce zig‑zag.  
**Adam / AdamW** — Adaptive per‑parameter step sizes; AdamW adds correct weight decay.  
**Learning‑rate schedules** — Warmup then cosine/step decay; second‑biggest lever after model choice.  
**Epoch / batch size** — Full pass; samples per step. Tune with LR for stability.  
**Early stopping** — Halt when validation stalls; avoids overfit and saves compute.  
**Gradient clipping** — Cap gradient norm to prevent explosions.

---

## Evaluation

**Accuracy** — Share correct; misleading with imbalance.  
**Precision / Recall** — Purity vs coverage of positives.  
**F1** — Harmonic mean of precision and recall.  
**ROC‑AUC** — Threshold‑free ranking quality; stable to class ratio.  
**PR‑AUC** — Better when positives are rare.  
**Confusion matrix** — TP/FP/FN/TN; pick threshold using business costs.  
**Log loss / Brier** — Penalize bad probabilities; supports calibration.  
**RMSE / MAE** — Regression error; MAE is outlier‑robust.  
**R^2** — Variance explained versus predicting the mean.  
**Calibration** — Do predicted probabilities match observed frequency? Use reliability curves or ECE.  
**Ranking metrics** — NDCG, MRR, hit rate for recommenders and search.  
**Clustering metrics** — Silhouette, ARI/AMI (when labels exist).  
**Domain metrics** — BLEU/ROUGE (NLP), WER (speech), IoU/mAP (vision).

---

## Time‑series primer (what you used above)

**Stationarity** — Stats don’t change over time. Use differencing or transformations to approximate.  
**Trend / seasonality** — Long‑term slope and periodic cycles.  
**ACF / PACF** — Autocorrelation and partial autocorrelation at lags; guide AR/MA orders.  
**ARIMA(p, d, q)** — Autoregression order p, differencing d, moving‑average q.  
**SARIMAX(p,d,q)(P,D,Q)_s** — Add seasonal terms (period s) and exogenous drivers X. Your scripts use (1,1,1)×(1,1,0)_12.  
**Holt‑Winters** — Exponential smoothing with level/trend/season; strong baseline.  
**Kalman / state‑space** — Latent‑state models; handle missing data naturally.  
**Backtesting** — Rolling/expanding windows that respect time order.  
**Horizon** — Steps ahead; uncertainty grows with horizon.  
**Exogenous variables** — External drivers (rates, CPI). In SARIMAX, pass as `exog`.  
**Drift** — Distribution/relationship shifts. Monitor and retrain on cadence.

**Helpful baselines to remember:**  
- Naïve seasonal: ŷ_{t+h} = y_{t+h−s}.  
- Drift model: ŷ_{t+h} = y_t + h * (y_t − y_{t−s}) / s.  
These often beat complicated models if the data is short or noisy.

---

## Probabilistic learning (more detail)

**MLE (maximum likelihood estimate)** — Choose parameters that maximize p(D|θ). Equivalent to minimizing NLL.  
**MAP (maximum a posteriori)** — Maximize p(θ|D) = p(D|θ)p(θ). L2 ≈ Gaussian prior; L1 ≈ Laplace prior.  
**Predictive intervals** — e.g., 90% PI = [lower, upper] quantiles of predictive distribution. Useful for capacity planning.  
**Uncertainty via ensembling** — Average predictions from differently seeded models to reduce variance; gives empirical uncertainty.  
**Bootstrap** — Resample with replacement; re‑fit; get sampling distribution of metrics or parameters.  
**KL divergence** — KL(P||Q) = sum P log(P/Q). Cross‑entropy = H(P,Q) = H(P) + KL(P||Q).  
**Calibration techniques** — Platt scaling, isotonic regression, temperature scaling for NNs.

---

## Causal inference and experimentation

**A/B test** — Randomized split; estimate causal effect with confidence intervals. Include power analysis and pre‑registered metrics.  
**ATE** — Average treatment effect = E[Y(1) − Y(0)].  
**Confounder** — Affects both treatment and outcome; adjust or risk bias.  
**DAGs** — Graph of assumptions; compute minimal adjustment sets.  
**Propensity score** — p(T=1|X); used for matching/weighting to balance covariates.  
**Instrumental variable** — Influences treatment but not outcome except through treatment.  
**Diff‑in‑diff / RDD** — Quasi‑experimental methods using time or thresholds.  
**Uplift modeling** — Predict individual treatment effect to target interventions efficiently.

---

## NLP and LLMs

**Tokenization** — Split text to tokens (BPE/WordPiece/SentencePiece).  
**Embeddings** — Dense vectors for tokens/sentences; cosine similarity powers search and RAG.  
**Positional encoding** — Inject order (sinusoidal, learned, rotary).  
**Context window** — Max tokens the model attends to; long‑context models change what you can do.  
**Decoding** — Greedy/beam (deterministic), top‑k/top‑p and temperature (stochastic).  
**Fine‑tuning / SFT** — Supervised adaptation to tasks.  
**LoRA / adapters** — Parameter‑efficient fine‑tuning by adding low‑rank matrices.  
**RLHF** — Train a reward model from human preferences; optimize responses with RL.  
**RAG** — Retrieve documents, then generate with citations; reduces hallucinations.  
**Hallucination** — Confident false output; mitigate with retrieval, constraints, and calibration.

---

## Vision and generative models

**Bounding box / segmentation** — Coarse vs pixel‑level localization.  
**Non‑maximum suppression (NMS)** — Deduplicate overlapping detections.  
**ResNet / U‑Net / ViT** — Common backbones in vision.  
**GAN** — Generator vs discriminator; sharp images; training instabilities.  
**VAE** — Probabilistic autoencoder with a latent space; enables interpolation.  
**Diffusion** — Learn to denoise; sampling walks from noise to image. Classifier‑free guidance (CFG) steers strength.

---

## Regularization and generalization

**Overfitting / underfitting** — Too complex vs too simple. Watch learning curves.  
**Bias‑variance trade‑off** — High bias → underfit; high variance → overfit.  
**Weight decay, dropout, early stopping** — Core regularizers.  
**Data augmentation, mixup, label smoothing** — Improve robustness and calibration.  
**Ensembling / stacking** — Average or meta‑learn multiple models to reduce variance.  
**Hyperparameter tuning** — Grid, random, Bayesian (Optuna). Always use a clean validation protocol.

---

## MLOps and production

**Pipeline** — Deterministic chain from raw data to predictions; make it reproducible.  
**Feature store** — Centralized, versioned features aligned for online/offline use.  
**Model registry** — Version models with metadata and approvals.  
**Experiment tracking** — Record params, metrics, artifacts (e.g., MLflow).  
**Serving** — Batch vs real‑time endpoints. Watch latency and cost.  
**Shadow/canary** — Route small traffic to new model safely.  
**Monitoring** — Data drift, concept drift, performance decay, cost, P95 latency.  
**Vector database** — ANN indices (FAISS/HNSW) for embeddings and RAG.

---

## Explainability, fairness, privacy

**SHAP / LIME** — Local feature attributions; sanity‑check drivers.  
**Counterfactuals** — Minimal input change to flip a prediction; actionable insight.  
**Fairness metrics** — Demographic parity, equalized odds, predictive parity; pick per use case.  
**Differential privacy** — Protect individuals with controlled noise (epsilon budget).  
**Federated learning** — Train across devices/tenants without centralizing raw data.  
**Model/data cards** — Document scope, risks, intended use, and metrics for stakeholders.

---

## Math quick references

**Vectors / matrices / tensors** — 1D/2D/3D+ arrays.  
**Norms** — L1: sum abs values; L2: sqrt(sum squares).  
**Gradient / Jacobian / Hessian** — First and higher derivatives.  
**Eigenvalues / eigenvectors; SVD** — Understand linear transforms and low‑rank structure.  
**Softmax** — softmax(z_i) = exp(z_i) / sum_j exp(z_j).  
**Convexity** — Single global optimum; easier optimization.  
**Information criteria** — AIC/BIC for model order selection (lower is better).

---

## Study plan suggestions

- Build a spaced‑repetition deck from the **bold terms**.  
- Re‑implement small pieces: standardization; logistic loss; ARIMA differencing; rolling backtest loop.  
- When forecasting, always compare to naïve seasonal and drift baselines.  
- When classifying, report PR‑AUC and calibration, not only accuracy.

---

## Minimal glossary index (for quick search)
Dataset; Sample; Feature; Label; Supervised; Unsupervised; Self‑supervised; Semi‑supervised; RL; Online; Batch; Transfer; Multitask; Few‑shot; Zero‑shot; Train/val/test; CV; Stratification; Leakage; Standardization; Normalization; One‑hot; Embedding; Imputation; Augmentation; Class imbalance; PCA; UMAP; t‑SNE; Linear; Logistic; Regularization; SVM; Tree; Random forest; Gradient boosting; k‑NN; Naive Bayes; k‑means; DBSCAN; GMM; HMM; VAR; VECM; MLP; CNN; RNN; LSTM; GRU; Attention; Transformer; Encoder; Decoder; Dropout; Batch norm; Layer norm; MSE; MAE; Cross‑entropy; SGD; Momentum; Adam; AdamW; LR schedule; Epoch; Batch size; Early stopping; Clipping; Accuracy; Precision; Recall; F1; ROC‑AUC; PR‑AUC; Confusion matrix; Log loss; Brier; RMSE; R^2; Calibration; NDCG; MRR; Silhouette; ARI; AMI; BLEU; ROUGE; WER; IoU; mAP; Stationarity; Trend; Seasonality; ACF; PACF; ARIMA; SARIMAX; Holt‑Winters; Kalman; Backtesting; Horizon; Exogenous; Drift; Likelihood; Prior; Posterior; MLE; MAP; Predictive distribution; Intervals; Ensemble; Bootstrap; KL; A/B; ATE; Confounder; DAG; Propensity; IV; DiD; RDD; Uplift; Tokenization; Embedding; Positional encoding; Context window; Decoding; SFT; LoRA; RLHF; RAG; Hallucination; NMS; ResNet; U‑Net; ViT; GAN; VAE; Diffusion; CFG; Overfitting; Bias‑variance; Weight decay; Early stopping; Mixup; Label smoothing; Ensembling; Hyperparameter search; Pipeline; Feature store; Model registry; Tracking; Serving; Shadow; Canary; Monitoring; Vector DB; SHAP; LIME; Counterfactual; Fairness; Differential privacy; Federated learning; Cards; Eigen/SVD; Softmax; Convexity; AIC/BIC.
