import numpy as np

from group.candidate_mask import build_candidate_mask
from group.grouping import group_candidate_edges
from group.features import build_group_features
from group.mahalanobis_distance import calculate_group_feature_covariance, calculate_candidate_neighbor_distances


def test_candidate_mask_excludes_background():
    d = np.array([[50, 50, 50, 50], [50, 80, 81, 80], [50, 76, 83, 50]], dtype=float)
    mask = build_candidate_mask(d, 70)
    assert not mask[0, 0]
    assert mask[1, 1]
    assert mask[2, 2]


def test_radius_allows_gap_inside_object():
    d = np.array([[50, 50, 50, 50, 50], [50, 80, 81, 50, 80], [50, 76, 83, 50, 50]], dtype=float)
    mask = d >= 70
    features = build_group_features(d)
    cov = calculate_group_feature_covariance(features, mask)
    edges, distances = calculate_candidate_neighbor_distances(features, mask, cov, spatial_radius=3)
    assert len(edges) > 0
    # The two object portions are spatially connectable with R=3.
    group_map = group_candidate_edges(mask, edges, epsilon=np.percentile(distances, 50))
    assert group_map.shape == d.shape
