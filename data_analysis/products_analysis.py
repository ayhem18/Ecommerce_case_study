"""
This script contains functionalities used to analyse the products
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF as smart_cdf
from typing import Tuple, List, Sequence

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

home = SCRIPT_DIR
current = home
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')
PREPARED_DATA_FOLDER = os.path.join(current, 'prepared_data')


def read_item_order_df() -> pd.DataFrame:
    item_order_csv = os.path.join(DATA_FOLDER, 'olist_order_items_dataset.csv')
    return pd.read_csv(item_order_csv)

def read_products_df() -> pd.DataFrame:
    products_csv_path = os.path.join(DATA_FOLDER, 'olist_products_dataset.csv')
    return pd.read_csv(products_csv_path).drop(columns=["product_length_cm", "product_height_cm", "product_width_cm"])


def prepare_item_orders(order_items_df: pd.DataFrame) -> pd.DataFrame:
    products_count = (order_items_df.groupby('product_id')['price'].agg(['count']). # computing how many times was each product bought
                                                            sort_values('count', ascending=False). # sort the products by the number of times it was bought
                                                            reset_index(). 
                                                            reset_index(). # call reset_index twice so that the 'index' will be added as a column to the dataframe
                                                            rename(columns={"index":"rank"})) # rename 'index' to 'rank'
    # let the ranks start from '1' instead of '0'
    products_count['rank'] = products_count['rank'] + 1

    # merge the 2 dataframes so we have the order information along with the popularity information
    return pd.merge(left=order_items_df, right=products_count, on='product_id', how='inner').sort_values('count', ascending=False)



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
        plt.axvline(x, ymax=y, color='c', label=f'top {100 * q}-th quantile popular products', c=color, linestyle='--')

    plt.xticks(np.linspace(1, np.max(ranks), 51), rotation=90)
    plt.yticks(np.linspace(0, 1, 21))
    plt.title("product market share: % of orders out of all orders !!! with respect to product popularity")
    plt.ylabel('product market share: %  of orders out of all orders !!!')
    plt.xlabel('product popularity rank')
    plt.legend()
    plt.show()

    # don't forget to return the quantiles: 
    return [int(x) for x in qx], qy

def filter_order_item_csv_top_products(order_item_count: pd.DataFrame, 
                             quantile_rank: int) -> pd.DataFrame:
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
    
    plt.hist(p0['count'], bins=20, density=False, label='1-photo products', color='b') # 'g' for green
    plt.xlabel('number of times a product was ordered')
    plt.ylabel('frequency')
    plt.xticks(ticks=[int(x) for x in np.linspace(p0['count'].min(), p0['count'].max(), 26)], rotation=90)
    plt.legend()
    plt.title('how many times are products with only 1 photo ordered ?')

    fig.add_subplot(1, 2, 2)
    plt.hist(p1['count'], bins=20, density=False, label='multiple-photo products', color='b') # 'g' for green
    plt.xlabel('number of times a product was ordered')
    plt.ylabel('frequency')
    plt.xticks(ticks=[int(x) for x in np.linspace(p1['count'].min(), p1['count'].max(), 26)], rotation=90)
    plt.legend()
    plt.title('how many times are products with multiple photo ordered ?')

    plt.show()

    return p0, p1
