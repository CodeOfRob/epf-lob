import joblib
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, norm
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from tqdm import tqdm


class StaticAsinh1Scaler(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible scaler implementing the 'asinh1' method.
    Includes persistence methods (save/load) for deployment in trading sims.
    """

    def __init__(self, features_to_scale, ttd_col, ttd_bins, epsilon=1e-8):
        self.features_to_scale = features_to_scale
        self.ttd_col = ttd_col
        self.ttd_bins = ttd_bins
        self.epsilon = np.float32(epsilon)
        self.profile_df_ = None
        self.mad_factor_ = 1 / norm.ppf(0.75)  # Approx. 1.4826

    def fit(self, X, y=None):
        """Learns the profiles from raw training data."""
        print("Fitting scaler...")
        df = X.copy()
        df['ttd_bin'] = pd.cut(df[self.ttd_col], bins=self.ttd_bins, right=False, labels=range(len(self.ttd_bins) - 1))

        profiles = []
        for feature in self.features_to_scale:
            if feature not in df.columns:
                warnings.warn(f"Feature not found in df to scale: {feature}")
                continue

            df[feature] = df[feature].astype(np.float32)

            grouped = df.groupby('ttd_bin', observed=False)[feature]
            median_profile = grouped.median()
            mad_profile = grouped.apply(lambda x: median_abs_deviation(x, nan_policy='omit')).astype(np.float32)

            feature_profile = pd.DataFrame({'median': median_profile, 'mad': mad_profile})
            feature_profile['feature'] = feature
            feature_profile = feature_profile.reset_index()
            profiles.append(feature_profile)

        self.profile_df_ = pd.concat(profiles, ignore_index=True)
        print("Scaler fitted.")
        return self

    def _merge_profiles(self, df):
        """Helper to merge the learned profiles onto a DataFrame."""
        df_out = df.copy()

        # 1. Create TTD bins on the data to be transformed
        df_out['ttd_bin'] = pd.cut(df_out[self.ttd_col], bins=self.ttd_bins, right=False,
                                   labels=range(len(self.ttd_bins) - 1))

        # 2. Prepare the profile_df for merging
        profile_pivot = self.profile_df_.pivot(index='ttd_bin', columns='feature', values=['median', 'mad'])

        # HIER WIRD DER MULTIINDEX AUFGELÖST
        # Erstelle eine einfache Spaltenliste: z.B. 'median_orderbook_slope'
        new_cols = {}
        for level1, level2 in profile_pivot.columns:
            new_cols[level1, level2] = f"{level1}_{level2}"
        profile_pivot.columns = profile_pivot.columns.map(new_cols)

        # 3. Merge the scaling parameters onto the dataframe
        # ACHTUNG: Der Merge-Key 'ttd_bin' muss im Input-DF als Spalte existieren
        df_merged = df_out.merge(
            profile_pivot,
            left_on='ttd_bin',
            right_index=True,  # Merging mit dem Index von profile_pivot
            how='left'
        )
        return df_merged

    def transform(self, X):
        """Applies transformation."""
        import pandas as pd

        if self.profile_df_ is None: raise RuntimeError("Scaler not fitted.")
        df_merged = self._merge_profiles(X)
        transformed_cols = {}
        for feature in tqdm(self.features_to_scale, "Transforming features with Asinh1Scaler"):
            if feature in X.columns:
                median_col = f"median_{feature}"
                mad_col = f"mad_{feature}"
                median_val = df_merged[median_col]
                mad_val = df_merged[mad_col]

                # 1. Normalize (set1)
                normalized_val = (X[feature].astype(np.float32) - median_val) / (
                        mad_val * np.float32(self.mad_factor_) + self.epsilon)
                # 2. Asinh
                transformed_cols[feature] = np.arcsinh(normalized_val).astype(np.float32)

        df_transformed = pd.DataFrame(transformed_cols, index=X.index)
        other_cols = [col for col in X.columns if col not in self.features_to_scale]
        return pd.concat([df_transformed, X[other_cols]], axis=1)

    def inverse_transform(self, X_transformed):
        """Reverses transformation."""

        if self.profile_df_ is None: raise RuntimeError("Scaler not fitted.")
        df_merged = self._merge_profiles(X_transformed)
        inverse_cols = {}
        for feature in tqdm(self.features_to_scale, "Inverse transforming features with Asinh1Scaler"):
            if feature in X_transformed.columns:
                median_col = f"median_{feature}"
                mad_col = f"mad_{feature}"
                median_val = df_merged[median_col]
                mad_val = df_merged[mad_col]
                # 1. Sinh
                descaled_asinh = np.sinh(X_transformed[feature].astype(np.float32)).astype(np.float32)
                # 2. Denormalize
                inverse_cols[feature] = (
                        (descaled_asinh * (mad_val * np.float32(self.mad_factor_) + self.epsilon)) + median_val).astype(
                    np.float32)

        df_inv = pd.DataFrame(inverse_cols, index=X_transformed.index)
        other_cols = [col for col in X_transformed.columns if col not in self.features_to_scale]
        return pd.concat([df_inv, X_transformed[other_cols]], axis=1)

    # --- PERSISTENCE METHODS ---

    def save(self, filepath):
        """
        Saves the fitted scaler to a file using joblib (binary).
        This is the standard, most robust way.
        """
        joblib.dump(self, filepath)
        print(f"Scaler saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Loads a fitted scaler from a joblib file.
        """
        scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return scaler
