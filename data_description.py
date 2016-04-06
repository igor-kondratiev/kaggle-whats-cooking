"""
Data description utils. Provides common information about data-set and it's properties
"""
from data_operations import extract_ingredients


def describe(df):
    """
    Return data-frame meta information
    @param df: data frame
    @type df: pandas.core.frame.DataFrame
    @return: data-frame meta information
    """
    ingredients = extract_ingredients(df)

    return (
        ('DataFrame shape', df.shape),
        ('Ingredients count', len(ingredients)),
    )
