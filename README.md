# Cancer-Model-Experiment-4
A Machine Learning Model to diagnose breast tumors as either Malignant (1) or Benign (0).


# Evaluating Sequential Boosting Algorithms (AdaBoost & GBM) in Oncology Classification #

## Project Overview & Clinical Objective ##
The objective of this phase was to determine if Boosting Algorithms (sequential ensemble models that learn from their own mistakes) could outperform our existing classification models on the Wisconsin Breast Cancer Dataset. Because this is a medical application, the strict success metric remained 100% Recall (0 False Negatives) while minimizing the False Positive (Panic Rate).

## The Benchmark to Beat ##
Our reigning champion was Logistic Regression. By utilizing an L1 (Lasso) penalty and class_weight='balanced', it mathematically reduced the 30-feature dataset down to just 2 "Golden Features" (perimeter_worst and concave points_worst).

- Benchmark Score: 100% Recall (0 Missed Cancers) with only 5 False Positives.

### Experiment 1: Baseline Boosting Performance ###
I deployed AdaBoost (which punishes mistakes using a weighted roulette wheel) and Gradient Boosting/GBM (which uses calculus to predict residual errors).
- AdaBoost Baseline: 4 Missed Cancers, 0 False Positives.
- GBM Baseline: 6 Missed Cancers, 0 False Positives.
- Observation: Both models were too conservative. GBM specifically suffered because scikit-learn's implementation lacks a class_weight parameter, rendering the algorithm "blind" to the medical urgency of avoiding False Negatives.

### Experiment 2: Threshold Hacking (The Panic Penalty) ###
To force AdaBoost to hit 0 Missed Cancers, I bypassed the default 50% probability threshold and ran a Threshold Scanner.
- The Result: I had to drop the threshold to a highly paranoid 35% to finally achieve 0 False Negatives.
- The Cost: The False Positive rate exploded to 55 false alarms.
- Conclusion: Forcing a tree-based boosting algorithm to hit perfect Recall via threshold manipulation caused an unacceptable level of clinical panic, disqualifying this approach.

### Experiment 3: The "Golden Features" Hybrid Hypothesis ###
- The Hypothesis: If AdaBoost and GBM were getting distracted by the 28 noisy features, I could improve them by feeding them only the 2 Golden Features discovered by Logistic Regression.
- The Untuned Result: Performance worsened. AdaBoost missed 5 cancers; GBM missed 9.
- The Tuned Result: I deployed GridSearchCV to smooth out the models in this 2D space. The tuning failed to improve AdaBoost (still 5 misses) and actually caused GBM to underfit (missing 10 cancers). The hypothesis was conclusively disproved.

## Key Findings & Lessons Learned ##
Through rigorous A/B testing and hyperparameter tuning, we uncovered three fundamental laws of Machine Learning geometry:
1. Feature Selection is Algorithm-Specific ("Apples to Oranges"): I proved that you cannot copy the optimal features from a Linear model and force them onto a Tree model. Trees need higher dimensions (more features) to build alternate routes around overlapping data. By stripping 28 features, I blinded the Boosting algorithms.
2. Stair-Steps vs. Smooth Diagonals: Boosting algorithms build complex "staircases" using vertical and horizontal splits. Human biology is messy and highly correlated. Logistic Regression won the tournament because its underlying math allows it to draw a smooth, continuous diagonal boundary, separating overlapping biological edge-cases far better than the rigid corners of a tree.
3. The Limits of Sequential Learning: While AdaBoost and GBM are brilliant at fixing their own errors, they are highly sensitive to noise and overlapping outliers. Without an intrinsic L1/L2 penalty to clean the data during training, they over-complicate their boundaries, leading to higher False Positive rates when forced to hit 100% Recall.
   
Final Verdict: Logistic Regression remains the undisputed champion. The standard Boosting algorithms (AdaBoost and GBM) were formally rejected for this specific clinical deployment.

### Resources ###
See codes [here](https://drive.google.com/file/d/1N80ckR6z-SxO1guolpVJD1kjyjuIEAN1/view?usp=drive_link)
