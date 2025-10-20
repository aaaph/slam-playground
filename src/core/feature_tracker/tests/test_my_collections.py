from core.feature_tracker.my_collections import ResettableDict


def test_my_collections():
    my_default_dict = {0: 0}
    my_resettable_dict = ResettableDict(my_default_dict)
    my_resettable_dict[0] = 1
    assert my_resettable_dict[0] == 1
    my_resettable_dict.clear()
    assert my_resettable_dict[0] == 0

    my_dict_with_set = {0: set()}
    my_resettable_dict_with_set = ResettableDict(my_dict_with_set)
    my_resettable_dict_with_set[0].add(1)
    assert my_resettable_dict_with_set[0] == {1}
    my_resettable_dict_with_set.clear()
    assert my_resettable_dict_with_set[0] == set()
