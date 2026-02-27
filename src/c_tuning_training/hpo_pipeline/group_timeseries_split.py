import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class GroupTimeSeriesSplit:
    """
    Time Series Split, der Gruppen respektiert.

    Anstatt Rohdaten-Indizes zu splitten, werden die EINZIGARTIGEN GRUPPEN (z.B. delivery_starts)
    chronologisch sortiert und DIESE dann mit TimeSeriesSplit aufgeteilt.

    Verhindert, dass Produkte (Stunden) in der Mitte durchgeschnitten werden.
    """

    def __init__(self, n_splits=5, max_train_size=None, gap=0):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """
        Erzeugt Indizes für Train/Test.

        :param groups: Muss übergeben werden! (z.B. df['delivery_start'])
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")

        # Wir stellen sicher, dass groups ein sauberes Numpy-Array oder Pandas-Series ist
        # und sortieren die einzigartigen Gruppen chronologisch.
        groups = np.array(groups)
        unique_groups = np.sort(np.unique(groups))

        # Wir nutzen den Standard-Splitter auf den GRUPPEN-IDs
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            gap=self.gap
        )

        # Jetzt iterieren wir über die Splits der Gruppen und mappen zurück auf die Zeilen
        for train_group_idxs, test_group_idxs in tscv.split(unique_groups):
            # Welche Gruppen-IDs (z.B. Timestamps) sind in Train/Test?
            train_groups = unique_groups[train_group_idxs]
            test_groups = unique_groups[test_group_idxs]

            # Finde die Zeilen im Original-Datensatz, die zu diesen Gruppen gehören
            # np.isin ist hier extrem effizient
            train_mask = np.isin(groups, train_groups)
            test_mask = np.isin(groups, test_groups)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
