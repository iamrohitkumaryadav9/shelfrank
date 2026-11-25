"""G5: facts about the product that do not depend on the query."""

import numpy as np
import pandas as pd

from shelfrank.data.product_features import parse_features

# Weight of the catalogue's mean rating, in ratings. A product with few ratings is pulled to it.
SHRINK = 10.0


def prior_features(pairs: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Columns g5_*. ``products`` rows are numbered by ``pi``."""
    ratings = products["rating_count"].astype(float).fillna(0.0)
    average = products["average_rating"].astype(float)
    known = average.notna() & (ratings > 0)
    mean = float(np.average(average[known], weights=ratings[known])) if known.any() else 0.0
    table = pd.DataFrame(
        {
            "g5_log_ratings": np.log1p(ratings),
            "g5_average_rating": average,
            "g5_shrunk_rating": (average.fillna(mean) * ratings + mean * SHRINK)
            / (ratings + SHRINK),
            "g5_log_reviews": np.log1p(products["review_count"].astype(float).fillna(0.0)),
            "g5_features": [len(parse_features(f).pairs) for f in products["product_features"]],
            "g5_has_description": products["product_description"].notna().astype(float),
        }
    )
    return table.iloc[pairs["pi"].to_numpy()].reset_index(drop=True)
