"""
This script contains functionalities used to analyse the products
"""

import os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF as smart_cdf
from typing import Tuple, List, Sequence

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

home = SCRIPT_DIR
current = home
while 'src' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')
PREPARED_DATA_FOLDER = os.path.join(current, 'prepared_data')


def read_item_order_df() -> pd.DataFrame:
    item_order_csv = os.path.join(DATA_FOLDER, 'olist_order_items_dataset.csv')
    return pd.read_csv(item_order_csv)

def read_products_df() -> pd.DataFrame:
    products_csv_path = os.path.join(DATA_FOLDER, 'olist_products_dataset.csv')
    product_df = pd.read_csv(products_csv_path).drop(columns=["product_length_cm", "product_height_cm", "product_width_cm"]).dropna()

    # make sure to translate the category names from portuguese to English
    from deep_translator import GoogleTranslator
    import re

    product_categories = np.unique(product_df['product_category_name'].dropna())
    
    # translate
    translator = GoogleTranslator(source='pt', target='en')
    categories_english = [translator.translate(text=re.sub('_', ' ', t)) for t in product_categories]

    product_df['product_category_name'] = product_df['product_category_name'].map(dict([(pt, en) for pt, en in zip(product_categories, categories_english)]))

    # modify the category names to consider only the first term
    product_df['product_category_name'] = product_df['product_category_name'].apply(lambda x: re.split(' ', x)[0])

    return product_df

def display_product_category_distribution(products_df: pd.DataFrame):
    cat_freq = products_df['product_category_name'].value_counts(normalize=True).sort_values(ascending=True)
    plt.figure(figsize=(15,6))
    plt.bar([re.split(" ", s)[0] for s in cat_freq.index], height=cat_freq.values * 100)
    plt.xticks(rotation=90)
    plt.yticks(np.linspace(0, 0.12, 11) * 100)
    plt.xlabel('Product Category')
    plt.ylabel('Percentage out of all products')
    plt.title('The number of products per product category')
    plt.show()

def prepare_item_orders(order_items_df: pd.DataFrame) -> pd.DataFrame:
    products_count = (order_items_df.groupby('product_id')['price'].agg(['count']). # computing how many times was each product bought
                                                            sort_values('count', ascending=False). # sort the products by the number of times it was bought
                                                            reset_index(). 
                                                            reset_index(). # call reset_index twice so that the 'index' will be added as a column to the dataframe
                                                            rename(columns={"index":"rank"})) # rename 'index' to 'rank'
    # let the ranks start from '1' instead of '0'
    products_count['rank'] = products_count['rank'] + 1

    # merge the 2 dataframes so we have the order information along with the popularity information
    return pd.merge(left=order_items_df, right=products_count, on='product_id', how='inner').sort_values(['count', 'rank'], ascending=[False, True])


def display_product_orders_skewness(order_item_count: pd.DataFrame, 
                                    figsize: Tuple[int, int]) -> List[int]:
    # set the figure size
    plt.figure(figsize=figsize)

    ranks = order_item_count['rank']
    # create the emperical cdf
    emperical_cdf = smart_cdf(ranks)
    # sample 1000 values from the 'ranks'
    sample = [int(x) for x in np.linspace(1, np.max(ranks), 1001)]

    # plot the emperical distribution
    plt.plot(sample, emperical_cdf(sample), label='emperical distribution')

    # calculate the 1-th 5-th and 10-th quantiles
    qx = np.quantile(np.unique(ranks), [0.01, 0.05, 0.1]) 
    qy = emperical_cdf([int(x) for x in qx])

    for x, y, q, color in zip(qx, qy, [0.01, 0.05, 0.1], ['r', 'b', 'g']):
        # add the vertical lines to the plot
        plt.axvline(x, ymax=y, color='c', label=f'top {int(100 * q)}-th quantile popular products', c=color, linestyle='--')

    plt.xticks(np.linspace(1, np.max(ranks), 51), rotation=90)
    plt.yticks(np.linspace(0, 1, 21))
    plt.title("product market share: % of orders out of all orders with respect to product popularity")
    plt.ylabel('product market share: %  of orders out of all orders !!!')
    plt.xlabel('product popularity rank')
    plt.legend()
    plt.show()

    # don't forget to return the quantiles: 
    return [int(x) for x in qx], qy


def filter_order_item_csv_top_products(order_item_count: pd.DataFrame, 
                             quantile_rank: int) -> pd.DataFrame:
    """This function filters the orders and products by popularity.

    Args:
        order_item_count (pd.DataFrame): the original dataframe with orders and products
        quantile_rank (int): The rank of the least popular product to keep
    Returns:
        pd.DataFrame: The new dataframe with orders only including top products 
    """
    
    if 'rank' not in order_item_count.columns: 
        raise ValueError(f"Please make sure the column 'rank' is present in the dataframe. Found columns: {order_item_count.columns}")  
    # keep only the top -quantile_rank products
    result = order_item_count[order_item_count['rank'] <= quantile_rank]    
    # save the resulting dataframe
    result.to_csv(os.path.join(PREPARED_DATA_FOLDER, 'order_item_top_products.csv'), index=False)

    return result


def prepare_products_df(top_products_order_item_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    ps = top_products_order_item_df.groupby('product_id')['price'].agg(['count']).sort_values(by='count', ascending=False)
    result = pd.merge(left=products_df, right=ps, how='inner', on='product_id').sort_values('count', ascending=False)

    # make sure to create the directory if needed
    os.makedirs(PREPARED_DATA_FOLDER, exist_ok=True)
    # save the result
    result.to_csv(os.path.join(PREPARED_DATA_FOLDER, 'top_products.csv'))
    return result


def analyze_products_num_photos(top_products: pd.DataFrame, figsize: Tuple[int, int]):
    top_products['product_photos_qty'].value_counts()
    # let's see if the number of photos make a difference 
    top_products['multi_photo'] = (top_products['product_photos_qty'] > 1).astype(int)
    p0, p1 = top_products[top_products['multi_photo'] == 0], top_products[top_products['multi_photo'] == 1]

    print(f"out of the top products, {len(p0)} have only 1 photo attached, {len(p1)} have at least 2 photos attached")

    fig = plt.figure(figsize=figsize)
    # the first plot will be the density function
    fig.add_subplot(1, 2, 1)
    
    plt.hist(p0['count'], bins=20, density=False, label='1-photo products') # 'g' for green
    plt.xlabel('number of times a product was ordered')
    plt.ylabel('frequency')
    plt.xticks(ticks=[int(x) for x in np.linspace(p0['count'].min(), p0['count'].max(), 26)], rotation=90)
    plt.legend()
    plt.title('how many times are products with only 1 photo ordered ?')

    fig.add_subplot(1, 2, 2)
    plt.hist(p1['count'], bins=20, density=False, label='multiple-photo products') # 'g' for green
    plt.xlabel('number of times a product was ordered')
    plt.ylabel('frequency')
    plt.xticks(ticks=[int(x) for x in np.linspace(p1['count'].min(), p1['count'].max(), 26)], rotation=90)
    plt.legend()
    plt.title('how many times are products with multiple photo ordered ?')

    plt.show()

    return p0, p1

