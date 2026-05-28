from crop.feature_groups import infer_feature_groups


def test_feature_groups_for_55_features():
    groups = infer_feature_groups(55)
    assert len(groups["global"]) == 5
    assert len(groups["node"]) == 6 + 32
    assert len(groups["topological"]) == 12
    assert len(groups["all"]) == 55
