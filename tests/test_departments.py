import pandas as pd

from shelfrank.data.departments import OTHER_DEPARTMENT, department_map, departments_of


def products(rows):
    return pd.DataFrame(rows, columns=["product_class", "category_hierarchy"])


def test_majority_first_level_wins():
    frame = products([("Beds", "F / a"), ("Beds", "F / b"), ("Beds", "K / a")])
    assert department_map(frame, min_products=1) == {"Beds": "F"}


def test_tie_is_alphabetical_and_order_free():
    frame = products([("Beds", "Z / a"), ("Beds", "A / a")])
    assert department_map(frame, 1) == department_map(frame.iloc[::-1], 1) == {"Beds": "A"}


def test_small_first_levels_become_other():
    frame = products([("Beds", "F / a"), ("Beds", "F / a"), ("Misc", "Sale / a")])
    assert department_map(frame, min_products=2) == {"Beds": "F", "Misc": OTHER_DEPARTMENT}


def test_multi_class_products_count_for_each_class():
    assert department_map(products([("Beds|Sofas", "F / a")]), 1) == {"Beds": "F", "Sofas": "F"}


def test_products_without_path_or_class_are_ignored():
    assert department_map(products([("Beds", None), (None, "F / a")]), 1) == {}


def test_missing_and_unknown_classes_are_other():
    assert departments_of(["Beds", float("nan"), "Ghost"], {"Beds": "F"}) == ["F", "Other", "Other"]
