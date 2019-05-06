import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_transactions():
    transactions_attributes = pd.read_csv("./transactions_attributes.csv") 
    return transactions_attributes
def read_transactions_products():
    data = pd.read_csv("./BlackFriday.csv") 
    transactions_product = pd.crosstab(data.User_ID,data.Product_ID).astype('bool').astype('int')
    print(transactions_product.sum(axis=0).median(axis = 0))
    print(transactions_product.sum(axis=0).quantile([0.25,0.5,0.75,0.9]))
    return transactions_product
# transactions_attributes = read_transactions()
# tid = transactions_attributes.drop(columns = ['User_ID'])
# frequent_predicate_set = apriori(tid, min_support=0.03, use_colnames=True)
# frequent_predicate_set.to_csv("frequent_predicate_set.csv")
transactions_product = read_transactions_products()
freq_itemsets = apriori(transactions_product, min_support=0.03, use_colnames=True)
freq_itemsets.to_csv("frequent_items_3.csv")

rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.0)
rules.to_csv("rules_freqitems_3.csv")
