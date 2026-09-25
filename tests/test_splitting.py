import numpy as np

from trajsig.evaluation import grouped_seed_splits


def test_grouped_seed_split_has_no_seed_overlap():
    seed_ids = [0, 1, 2, 0, 1, 2]
    for train, test in grouped_seed_splits(seed_ids):
        train_seeds = set(np.asarray(seed_ids)[train])
        test_seeds = set(np.asarray(seed_ids)[test])
        assert train_seeds.isdisjoint(test_seeds)
        assert len(test_seeds) == 1

