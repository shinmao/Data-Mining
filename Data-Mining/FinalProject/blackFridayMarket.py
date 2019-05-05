import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_transactions():
    transactions_productCat = pd.read_csv("./transactions_productCat.csv") 
    return transactions_productCat
transactions_productCat = read_transactions()
frequent_itemsets = apriori(transactions_productCat, min_support=0.6, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules)
