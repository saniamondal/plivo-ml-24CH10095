# NOTES

1. The initial feature set suggested by the assistant used absolute F0 values — the human immediately flagged this would fail on Hindi speakers whose pitch range differs, and forced a switch to normalised relative trajectories (slope ratios, not raw Hz), which is what made cross-lingual transfer actually work.
2. The human noticed the first OOF score (1600ms, baseline-equivalent) was because the model was fitting on in-turn pauses without grouping by turn_id — they insisted on GroupKFold by turn_id, which fixed the validation and revealed the real signal.
3. Energy decay and final-syllable lengthening were the human's explicit linguistic intuitions: "statements fall in energy and stretch the last syllable" — the AI implemented these only after the human articulated the phonetic reasoning.
4. The human caught that the original cumulative silence feature was accidentally using the current pause's own duration (leakage), and corrected it to sum only prior pauses (pause_index < current).
5. GradientBoostingClassifier was chosen after the human observed that decision boundaries for prosodic cues are non-linear and shallow-tree ensembles capture them better than logistic regression at this data size.
6. The human explicitly required a cold holdout test (Check 5) after suspecting the 100ms in-sample score was inflated — this revealed the honest generalisation (800ms on 20 unseen Hindi turns) and prevented overconfidence.
7. Isotonic calibration was added because the human observed predicted probabilities were poorly spread (most clustered near 0 or 1), making the sweep over thresholds brittle — calibration widened the useful operating range.
8. The human required all five leakage checks to be run in order and reviewed the grep output line-by-line, catching that pause_end appeared in scorer code (safe) not feature code — a distinction the automated check initially got wrong.
9. The decision to train jointly on English and Hindi (rather than English-only) came from the human reading the assignment brief carefully: "hidden test is mostly Hindi" — the AI had initially planned English-only training.
10. Final operating point (threshold=0.25, delay=800ms on Hindi holdout) was selected by the human after reviewing the sweep table; the human preferred a slightly lower threshold to fire earlier on obvious EOTs rather than waiting for certainty.
