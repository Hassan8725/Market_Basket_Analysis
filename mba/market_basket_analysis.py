
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def load_data(filepath):
    """Load the market basket dataset from a given filepath."""
    return pd.read_csv(filepath, header=None)

def prepare_transactions(df):
    """Convert the DataFrame into a list of transactions."""
    transactions = []
    for i in range(len(df)):
        transactions.append([str(df.values[i, j]) for j in range(df.shape[1]) if str(df.values[i, j]) != 'nan'])
    return transactions

def encode_transactions(transactions):
    """Encode the list of transactions into a one-hot encoded DataFrame."""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

def perform_apriori(df, min_support=0.01):
    """Perform the Apriori algorithm on the given DataFrame."""
    return apriori(df, min_support=min_support, use_colnames=True)

def generate_rules(frequent_itemsets, metric="lift", min_threshold=1):
    """Generate association rules from frequent itemsets."""
    return association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

# # Sample usage of the functions
# if __name__ == "__main__":
#     # Replace with the actual filepath
#     filepath = 'path_to_dataset.csv'
#     df = load_data(filepath)
#     transactions = prepare_transactions(df)
#     df_encoded = encode_transactions(transactions)
#     frequent_itemsets = perform_apriori(df_encoded)
#     rules = generate_rules(frequent_itemsets)
