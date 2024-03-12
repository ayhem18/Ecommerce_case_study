"""
This script contains the functionalities to analyze the sellers data
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from typing import List, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent
DATA_FOLDER = os.path.join(current, 'data')


def read_order_items_csv():
    item_order_csv = os.path.join(DATA_FOLDER, 'olist_order_items_dataset.csv')
    return pd.read_csv(item_order_csv)

def read_seller_csv():
    sellers_csv_path = os.path.join(DATA_FOLDER, 'olist_sellers_dataset.csv')
    return pd.read_csv(sellers_csv_path)


def analyze_product_per_seller(order_item_df: pd.DataFrame, display: bool = True) -> List[str]:
    # this line groups the orders by their products and for each product calculate the number of unique sellers who sell the given product
    # afterwards it sorts them descendingly according to this number
    product_by_seller = order_item_df.groupby('product_id')['seller_id'].agg(lambda x: len(np.unique(x))).sort_values(ascending=False)    
    # extract the id of products sold by at least 2 sellers
    multi_seller_products = (product_by_seller[product_by_seller > 1]).index.tolist()
    
    if display:
        print(f"The total number of products sold: {len(product_by_seller)}")
        print(f"The number of products sold by at least 2 sellers: {len(multi_seller_products)}")
        print(f"That's only: {round(100 * len(multi_seller_products) / len(product_by_seller), 4)} %")

    return multi_seller_products


def prepare_multi_seller_price_data(order_item_df: pd.DataFrame,
                                    sellers_df: pd.DataFrame,
                                    multi_seller_products: List[str]) -> pd.DataFrame:
    # the first step is to filter the products that are sold by multiple sellers
    multi_seller_orders = order_item_df[order_item_df['product_id'].isin(multi_seller_products)]
    # merge the data of the orders with that of the sellers
    multi_seller_orders = pd.merge(left=multi_seller_orders, right=sellers_df, how='inner', on='seller_id')
    # keep only the useful columns
    multi_seller_orders = multi_seller_orders[['order_id', 'product_id', 'seller_id', 'price', 'freight_value', 'seller_state']]
    return multi_seller_orders


def average_product_price_by_state(multi_seller_orders_df: pd.DataFrame,
                                   state1: str, 
                                   state2: str) -> pd.DataFrame:
    
    # extract the list of product ids that 
    pids = multi_seller_orders_df['product_id'].unique()
    
    product_average_prices = defaultdict(lambda: {})
    
    # for each product
    for pi in pids: 
        # consider only the orders including this specific product and sellers belonging to the given states 
        product_sellers = multi_seller_orders_df[(multi_seller_orders_df['product_id'] == pi) & (multi_seller_orders_df['seller_state'].isin([state1, state2]))]
        # group by state and consider the average price suggested by sellers from different states
        average_prices_by_state = product_sellers.groupby('seller_state')['price'].agg(['mean'])
        
        # only add products that are sold by both sellers
        if sorted([state1, state2]) != average_prices_by_state.index.tolist():
            continue

        product_average_prices[pi][state1] = average_prices_by_state.loc[state1].item()
        product_average_prices[pi][state2] = average_prices_by_state.loc[state2].item()

    # convert the results into a dataframe
    product_average_prices = [{"product_id": k, state1: v[state1], state2: v[state2]} for k, v in product_average_prices.items()]
    return pd.DataFrame(product_average_prices)


def display_average_price_distribution(s1_ap: Sequence[float],
                                       s2_ap: Sequence[float], 
                                       state1: str,
                                       state2: str,
                                       log_scale: bool,
                                       figsize: Tuple[int, int] = None):
    # apply the log transform if needed
    if log_scale:
        s1_ap, s2_ap = np.log2(s1_ap), np.log2(s2_ap)

    figsize = (15, 8) if figsize is None else figsize
    fig = plt.figure(figsize=figsize) 


    # plotting the first distribution
    fig.add_subplot(1, 3, 1)
    sample_frequncy = np.max(np.histogram(s1_ap)[0])
    
    plt.hist(s1_ap, bins=20)
    plt.xlabel(f'average price {"log_scale" if log_scale else ""}')
    plt.ylabel('frequency')
    plt.xticks(ticks=np.linspace(np.min(s1_ap), np.max(s1_ap), 26), rotation=90)
    plt.yticks([int(x) for x in np.linspace(1, sample_frequncy, 11)])
    plt.title(f'The distribution of average prices in {state1}')

    # plotting the 2nd distribution
    fig.add_subplot(1, 3, 2)
    sample_frequncy = np.max(np.histogram(s2_ap)[0])
    plt.hist(s2_ap, bins=20)
    plt.xlabel(f'average price {"log_scale" if log_scale else ""}')
    plt.ylabel('frequency')
    plt.xticks(ticks=np.linspace(np.min(s2_ap), np.max(s2_ap), 26), rotation=90)
    plt.yticks([int(x) for x in np.linspace(1, sample_frequncy, 11)])
    plt.title(f'The distribution of average prices in {state2}')
    plt.show()
