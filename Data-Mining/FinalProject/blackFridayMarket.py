import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_transactions():
    transactions_attributes = pd.read_csv("./transactions_attributes.csv") 
    return transactions_attributes
transactions_attributes = read_transactions()
tid = transactions_attributes = transactions_attributes.drop(columns = ['User_ID'])
frequent_predicate_set = apriori(tid, min_support=0.03, use_colnames=True)
frequent_predicate_set.to_csv("frequent_predicate_set.csv")

rules = association_rules(frequent_predicate_set, metric="lift", min_threshold=1.0)
rules.to_csv("rules.csv")
