import pandas as pd 
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_transactions():
    transactions_productCat = pd.read_csv("./transactions_productCat.csv") 
    return transactions_productCat

def user_attri(blackfriday, attribute, user_id):
    agg_functions = {'Gender':'first','Age':'first','Occupation':'first', 'City_Category':'first', 'Stay_In_Current_City_Years':'first', 'Marital_Status':'first'}
    df = blackfriday.drop(columns=['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']).groupby(blackfriday['User_ID']).agg(agg_functions)
    return df

def freq_with_attribute(blackfriday, freq_item):
    attribute = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    user_data = [[0] for i in range(5891)]
    user_id = []
    data = pd.read_csv('./transactions.csv')
    #print(data['1'][5890])
    # user_data: product category transactions of user
    for i in range(5891):
        profile = []
        for j in range(18):
            if data[str(j+1)][i] == 1:
                profile.append(str(j+1))
        user_data[i] = profile
    # all unique user id
    for i in range(537577):
        x = blackfriday['User_ID'][i]
        if x not in user_id:
            user_id.append(blackfriday['User_ID'][i])
    df = user_attri(blackfriday, attribute, user_id)
    freq_attri = []
    for i in range(len(df)):
        for j in range(len(freq_item)):
            freq_profile = []
            if user_data[i] == sorted(list(freq_item[j])):
                freq_profile.append(sorted(list(freq_item[j])))
                tmp1 = user_id[i]
                for k in range(len(attribute)):
                    tmp2 = attribute[k]
                    freq_profile.append(df[tmp2][tmp1])
                freq_attri.append(freq_profile)
    return freq_attri

def create_csv(freq_attri):
    df = pd.DataFrame(np.array(freq_attri), columns=['freq_itemset', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status'])
    df.to_csv('frequent_set_with_attribute.csv', index=False)
    
def main():
    data = pd.read_csv("./BlackFriday.csv")
    transactions_productCat = read_transactions()
    frequent_itemsets = apriori(transactions_productCat, min_support=0.6, use_colnames=True)
    freq_item = frequent_itemsets['itemsets'].tolist()
    #print(freq_item)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    freq_attri = freq_with_attribute(data, freq_item)
    create_csv(freq_attri)

if __name__ == '__main__':
    main()