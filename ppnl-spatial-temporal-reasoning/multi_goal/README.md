This directory contains datasets for multi-goal path planning. 

Under each sub-directory, you can find a ``json`` file for each of the following splits:

- ``n_goals_train_set_6x6_samples.json``: training set (6x6 environments, 1-5 obstacles).
- ``n_goals_dev_set_6x6_samples.json``: development set for 6x6 environments with ``n`` goals.
- ``n_goals_test_seen_6x6_samples.json``: 6x6 seen test set (same obstacle arrangments as the training set, but with different initial location/goal placements) for ``n``-goal path planning.
- ``n_goals_unseen_6x6_samples.json``: 6x6  test set with different obstacle arrangments as the training set, but with different initial location/goal placements for ``n``-goal path planning. 
- ``n_goals_test_seen_5x5_samples.json``: 5x5 test set for ``n``-goal path planning problems.
- ``n_goals_test_seen_7x7_samples.json``: 7x7 test set for ``n``-goal path planning problems. 
-  ``n_goals_test_unseen_6x6more_obstacles_samples.json``: OOD test-set for environments consisiting of more obstacles (6-11) than the training environments (1-5). 