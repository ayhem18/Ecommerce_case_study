"""
This script contains code for analysing the customer base
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent
DATA_FOLDER = os.path.join(current, 'data')


def read_customer_data() -> pd.DataFrame:
    customers_csv_path = os.path.join(DATA_FOLDER, 'olist_customers_dataset.csv')
    return pd.read_csv(customers_csv_path)

def unique_elements_per_column_customers(customers_df: pd.DataFrame) -> None:
    # let's count the unqiue values for each column:
    for col_name in customers_df.columns: 
        print("#" * 10)
        print(f"the column: '{col_name}' column has {len(customers_df[col_name].unique())} unique values")    

def customers_repeated_purchases(customers_df: pd.DataFrame) -> None:
    """This function describes the percentage of customers who used the platform more than once
    Args:
        customers_df (pd.DataFrame): The dataframe with customer information
    """
    print("Let's consider customers that used the platform multiple times")
    customers_counts = customers_df.groupby('customer_unique_id')['customer_state'].agg(['count'])
    customers_count1 = customers_counts[customers_counts['count'] > 1].sort_values(by='count', ascending=False)
    customers_count4 = customers_counts[customers_counts['count'] > 4].sort_values(by='count', ascending=False)

    total_customer_count = len(customers_df['customer_unique_id'].unique())
    
    print(f"out of {total_customer_count} customers, only {len(customers_count1)} ordered at least 2 times:" 
          f"{round(100 * len(customers_count1) / total_customer_count, 4)} % of the customer base")

    print(f"out of {total_customer_count} customers, only {len(customers_count4)} ordered at least 4 times:" 
          f"{round(100 * len(customers_count4) / total_customer_count, 4)} % of the customer base")


def get_cool_customers(customers_df: pd.DataFrame, min_repeated_purchases: int = 2) -> pd.DataFrame:
    if min_repeated_purchases < 2:
        raise ValueError(f"cool customers order at least twice, found repeated purchases: {min_repeated_purchases}")
    customers_counts = customers_df.groupby('customer_unique_id')['customer_state'].agg(['count'])
    customers_counts = customers_counts[customers_counts['count'] >= min_repeated_purchases].sort_values(by='count', ascending=False)
    return pd.merge(left=customers_df, right=customers_counts, on='customer_unique_id', how='inner').sort_values(by='count', ascending=False)


def visualize_customers_distribution_per_state(customer_df: pd.DataFrame, figsize: Tuple[int, int] = None):
    """a plain visualization of the distribution of customers with respect to the state
    """
    if figsize is None:
        figsize = (10, 6)
    
    plt.figure(figsize=figsize)
    states = customer_df['customer_state'].value_counts(normalize=True)
    
    plt.bar(list(states.index), height=states.values)
    plt.xticks(rotation=45)
    
    plt.xlabel("Customer State")
    plt.ylabel("Percentage Of Customers Out Of All Customers")
    
    plt.title("Percentage Of Customers Per State")
    plt.show()

def visualize_cool_customers_distribution_per_state(cool_customers: pd.DataFrame, figsize: Tuple[int, int]) -> None:
    """
    a plain visualization of the distribution of cool customers (those who use the platform at least twice) with respect to the different states
    """
    if figsize is None:
        figsize = (10, 6)

    plt.figure(figsize=figsize)
    states = cool_customers['customer_state'].value_counts(normalize=True)

    plt.bar(list(states.index), height=states.values)
    plt.xticks(rotation=45)
    
    plt.xlabel("Cool Customer State")
    plt.ylabel("Percentage Of Cool Customers Out Of All Cool Customers")
    
    plt.title("Percentage Of Cool Customers Per State")
    plt.show()

