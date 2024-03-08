"""
This script contains the functionalities to analyze the sellers data
"""
import os

import numpy as np
import pandas as pd

from pathlib import Path
from collections import defaultdict
from typing import Dict, List


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
        
        # it is worth warning the user when we don't find the states we are looking for
        if sorted([state1, state2]) != average_prices_by_state.index.tolist():
            # print(f"it seems that product with id {pi} does not have sellers from both {state1} and {state2}")
            # only save the data when both states are present
            continue

        product_average_prices[pi][state1] = average_prices_by_state.loc[state1].item()
        product_average_prices[pi][state2] = average_prices_by_state.loc[state2].item()

    # convert the result into a dataframe
    product_average_prices = [{"product_id": k, state1: v[state1], state2: v[state2]} for k, v in product_average_prices.items()]
    return pd.DataFrame(product_average_prices)

if __name__ == '__main__':
    sellers_df = read_seller_csv()
    order_items_df = read_order_items_csv()
    multi_seller_products = analyze_product_per_seller(order_item_df=order_items_df, display=False)
    multi_seller_orders = prepare_multi_seller_price_data(order_item_df=order_items_df, 
                                                          sellers_df=sellers_df, 
                                                          multi_seller_products=multi_seller_products)

    sp_rj_average_product_price = average_product_price_by_state(multi_seller_orders_df=multi_seller_orders,
                                                                    state1="SP", 
                                                                    state2="RJ")

    sp_ap, rj_ap = np.log2(sp_rj_average_product_price['SP']), np.log2(sp_rj_average_product_price['RJ'])

    from stats_utils import test_sample_normal_distribution
    pvalues = test_sample_normal_distribution(sample=sp_ap) 
    print(pvalues)
