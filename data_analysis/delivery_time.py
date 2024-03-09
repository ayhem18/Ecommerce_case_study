import os
from pathlib import Path
home = os.getcwd()
current = home
while 'mid' not in os.listdir(current):
    current = Path(current).parent
DATA_FOLDER = os.path.join(current, 'mid')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def calculte_delivery_time(df):
    df["delivery_time"] = pd.to_datetime(df["order_delivered_customer_date"]) - pd.to_datetime(df["order_purchase_timestamp"])
    df["delivery_time"] = df["delivery_time"].dt.total_seconds() / 3600
    return df

def average_delivery_time_by_location(data):
    # Calculate the average delivery time for each seller
    avg_delivery_time_by_state= data.groupby('seller_state')['delivery_time'].mean().reset_index().sort_values(by="delivery_time")

    # Create a scatter plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='seller_state', y='delivery_time', data=avg_delivery_time_by_state, color='blue')
    plt.title('Average Delivery Time by Seller Location')
    plt.xlabel('Seller Location (State)')
    plt.ylabel('Average Delivery Time (hours)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()

def state_sellers_delivery_times(data, state):
    df = data[data['seller_state'] == state].groupby('seller_id')['delivery_time'].mean().reset_index()
    df.rename(columns={'delivery_time': 'average_delivery_time'}, inplace=True)

    return df

from stats_utils import remove_outliers
def compare_two_cities(data, city_one, city_two, ro_one=False, ro_two=False):
    

    df_one = state_sellers_delivery_times(data, city_one)["average_delivery_time"].dropna().sort_values()
    df_two = state_sellers_delivery_times(data, city_two)["average_delivery_time"].dropna().sort_values()

    if ro_one : df_one = remove_outliers(df_one)
    if ro_two : df_two = remove_outliers(df_two)

    from scipy.stats import ttest_ind

    # Perform the two-sample t-test
    t_statistic, p_value = ttest_ind(df_one, df_two, equal_var=False)

    if(np.isnan(p_value)): return

    print("Two-Sample T-Test Results:")
    print(f"T-Statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    if p_value > 0.05:
        print("Fail to reject the null hypothesis. There is no significant difference between the means.")
    else:
        print("Reject the null hypothesis. There is a significant difference between the means.")
    

# read from files
sellers = pd.read_csv(os.path.join(DATA_FOLDER, 'olist_sellers_dataset.csv'))
order_items = pd.read_csv(os.path.join(DATA_FOLDER, 'order_item_top_products.csv'))
orders = pd.read_csv(os.path.join(DATA_FOLDER, 'olist_orders_dataset.csv'))

# merge dataframes
merged_data = pd.merge(sellers, order_items, on='seller_id')
merged_data = pd.merge(merged_data, orders, on='order_id')

data = calculte_delivery_time(merged_data)

# two cities with the biggest sellers count
print("SP and PR")
compare_two_cities(data, "SP", "PR", ro_one=True)

print("PR and MG")
compare_two_cities(data, "MG", "PR", ro_one=True, ro_two=True)

"""
seller_state
ES      1
MA      1
MS      1
MT      1
RN      1
CE      2
PE      2
BA      3
DF      7
GO      7
RS     19
SC     35
RJ     43
MG     63
PR     70
SP    419
"""