"""
This script contains the functionalities used to analyze the payments data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Sequence, List
from stats_utils import remove_outliers

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent
DATA_FOLDER = os.path.join(current, 'data')


def read_payments_csv():
    payments_csv_path = os.path.join(DATA_FOLDER, 'olist_order_payments_dataset.csv')
    payments = pd.read_csv(payments_csv_path)[['order_id', 'payment_value']]
    p = payments.groupby('order_id')['payment_value'].agg(['sum']).rename(columns={"sum": "pay"})
    return p

def read_orders_csv():
    orders_csv_path = os.path.join(DATA_FOLDER, 'olist_orders_dataset.csv')
    orders = pd.read_csv(orders_csv_path)
    return orders[['order_id', 'customer_id']]
    
def aggregate_spendings_by_state(orders_df: pd.DataFrame, payments_df: pd.DataFrame, customers_df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    left = pd.merge(left=orders_df, right=payments_df, on='order_id', how='inner')
    right = customers_df[['customer_id', 'customer_state']]
    order_customer_pay = pd.merge(left=left, right=right, on='customer_id', how='inner')
    order_customer_pay.head()

    customer_pay = order_customer_pay[['pay', 'customer_state']]
    # let's consider the average spending of 2 thirds of the customers residing in the largest 3 states in Brazil !!!
    sp_pay = customer_pay[customer_pay['customer_state'] == 'SP']['pay']
    mg_pay = customer_pay[customer_pay['customer_state'] == 'MG']['pay']
    rj_pay = customer_pay[customer_pay['customer_state'] == 'RJ']['pay']

    return order_customer_pay, sp_pay, mg_pay, rj_pay

def analyze_average_spending_per_state(sp_pay: Sequence, 
                                       mg_pay: Sequence, 
                                       rj_pay: Sequence, 
                                       figsize: Tuple[int, int] = None) -> Tuple[List[float], List[float], List[float]]:

    if figsize is None:
        figsize = (20, 10)
    # TODO: improve the display !!
    # the initial data contains several outliers, let's remove them 
    sp_pay_filtered = remove_outliers(sp_pay)
    mg_pay_filtered = remove_outliers(mg_pay)
    rj_pay_filtered = remove_outliers(rj_pay)

    # let's display the distribution
    fig = plt.figure(figsize=figsize) 
    fig.add_subplot(1, 3, 1)
    plt.hist(sp_pay_filtered, bins=20)
    plt.title('SAO PAULO')

    fig.add_subplot(1, 3, 2)
    plt.hist(mg_pay_filtered, bins=20)
    plt.title('MG')

    fig.add_subplot(1, 3, 3)
    plt.hist(rj_pay_filtered, bins=20)
    plt.title("RIO DE JENEIRO")
    plt.show()

    return sp_pay_filtered, mg_pay_filtered, rj_pay_filtered

def box_plot_spending_by_state(sp_pay_filtered: List[float], 
                               mg_pay_filtered: List[float], 
                               rj_pay_filtered: List[float]):
    # TODO: the visualization is minimalistic to say the least ...
    plt.boxplot([sp_pay_filtered, mg_pay_filtered, rj_pay_filtered])
    plt.title('customers spending across states (after outlier removal)')
    plt.show()


if __name__ == '__main__':
    p = read_payments_csv()
    print(p.head())

