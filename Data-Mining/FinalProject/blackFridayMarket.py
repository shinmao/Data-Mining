import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_transactions():
    transactions_attributes = pd.read_csv("./transactions_attributes.csv") 
    return transactions_attributes
def read_transactions_products():
    data = pd.read_csv("./BlackFriday.csv") 
    product = prepend(sorted(data.Product_ID.unique()),"Product = ")
    transactions_product = pd.crosstab(data.User_ID,data.Product_ID).astype('bool').astype('int')
    colnames = product
    transactions_product.columns = colnames
    print(transactions_product)
    print(transactions_product.sum(axis=0).median(axis = 0))
    print(transactions_product.sum(axis=0).quantile([0.25,0.5,0.75,0.9]))
    return transactions_product,data

def prepend(list, str): 
      
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

def products_gender(transactions_product,data):
    # obtain the unique values of gender attribute. Can be used for the column later
    gender = prepend(sorted(data.Gender.unique()),"Gender = ")
    # binarize the gender attribute: F & M
    genders = pd.crosstab(data.User_ID,data.Gender).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,genders], axis=1, sort=False)
    colnames = list(transactions_product.columns) + gender
    transactions.columns = colnames
    # obtain frequent item sets via apriori algotithm; including the gender attribute
    # set min support as 0.03 because the third quantile number of transactions of the product is 192
    # 0.25: 19.0; 0.50: 70.0; 0.75: 192.0; 0.90: 396.8
    # meaning that we only care about the top 25% of popular products; so the min support is 192 / 5892 = 0.03
    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_g.csv")
    # generate the association rules by filtering the metric lift greater than 2.0
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Gender" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Gender")]
    # eliminate rules that contain attribute "Gender" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Gender")]
    # rules.to_csv("rules_freqitems_3_g.csv")
    return rules

def products_age(transactions_product,data):
    age = prepend(sorted(data.Age.unique()),"Age = ")
    ages = pd.crosstab(data.User_ID,data.Age).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,ages], axis=1, sort=False)
    colnames = list(transactions_product.columns) + age
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_age.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Age" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Age")]
    # eliminate rules that contain attribute "Age" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Age")]

    # rules.to_csv("rules_freqitems_3_age.csv")
    return rules

def products_occupation(transactions_product,data):
    occupation = prepend(sorted(data.Occupation.unique()),"Occupation = ")
    occupations = pd.crosstab(data.User_ID,data.Occupation).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,occupations], axis=1, sort=False)
    colnames = list(transactions_product.columns) + occupation
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_occ.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Occupation" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Occupation")]
    # eliminate rules that contain attribute "Occupation" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Occupation")]
    # rules.to_csv("rules_freqitems_3_occ.csv")
    return rules

def products_city(transactions_product,data):
    city = prepend(sorted(data.City_Category.unique()),"City = ")
    cities = pd.crosstab(data.User_ID,data.City_Category).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,cities], axis=1, sort=False)
    colnames = list(transactions_product.columns) + city
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_city.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "City" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("City")]
    # eliminate rules that contain attribute "City" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("City")]
    # rules.to_csv("rules_freqitems_3_city.csv")
    return rules


def products_city_stay(transactions_product,data):
    city_s = prepend(sorted(data.Stay_In_Current_City_Years.unique()),"Stay_In_Current_City_Years = ")
    city_stay = pd.crosstab(data.User_ID,data.Stay_In_Current_City_Years).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,city_stay], axis=1, sort=False)
    colnames = list(transactions_product.columns) + city_s
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_city_stay.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Stay_In_Current_City_Years" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Stay_In_Current_City_Years")]
    # eliminate rules that contain attribute "Stay_In_Current_City_Years" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Stay_In_Current_City_Years")]
    # rules.to_csv("rules_freqitems_3_city_stay.csv")
    return rules

def products_marital(transactions_product,data):
    marital_s = prepend(sorted(data.Marital_Status.unique()),"Marital_Status = ")
    marital = pd.crosstab(data.User_ID,data.Marital_Status).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,marital], axis=1, sort=False)
    colnames = list(transactions_product.columns) + marital_s
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_marital.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Marital_Status" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Marital_Status")]
    # eliminate rules that contain attribute "Marital_Status" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Marital_Status")]

    # rules.to_csv("rules_freqitems_3_marital.csv")
    return rules


# transactions_attributes = read_transactions()
# tid = transactions_attributes.drop(columns = ['User_ID'])
# frequent_predicate_set = apriori(tid, min_support=0.03, use_colnames=True)
# frequent_predicate_set.to_csv("frequent_predicate_set.csv")
# rules = association_rules(frequent_predicate_set, metric="lift", min_threshold=1.0)
# rules.to_csv("rule_freq_predicate.csv")

transactions_product, data = read_transactions_products()
rules_g = products_gender(transactions_product,data)
rules_age = products_age(transactions_product,data)
rules_occ = products_occupation(transactions_product,data)
rules_city = products_city(transactions_product,data)
rules_city_s = products_city_stay(transactions_product,data)
rules_marital = products_marital(transactions_product,data)
