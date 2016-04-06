"""
Module with data operations definitions
"""


def extract_ingredients(df):
    """
    Extract ingredients list from data-frame
    @param df: source data-frame
    @type df: pandas.core.frame.DataFrame
    @return: ingredients list
    @rtype: list
    """
    ingredients = set()
    for index, row in df.iterrows():
        ingredients.update(row['ingredients'])

    ingredients = list(ingredients)
    ingredients.sort()
    return ingredients


def extract_cuisines(df):
    """
    Extract cuisines list from data-frame
    @param df: source data-frame
    @type df: pandas.core.frame.DataFrame
    @return: cuisines list
    @rtype: list
    """
    cuisines = set()
    for index, row in df.iterrows():
        cuisines.add(row['cuisine'])

    cuisines = list(cuisines)
    cuisines.sort()
    return cuisines
