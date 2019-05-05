import pandas as pd 
def read_data():
    data = pd.read_csv("./BlackFriday.csv") 
    return data

def create_transactions_productCat(data):
    transactions = pd.crosstab(data.User_ID, data.Product_Category_1).astype('bool').astype('int')
    # use transactions.iloc[:,0:5] to access entries
    transactions.to_csv("transactions_productCat.csv",index = False)
data = read_data()
create_transactions_productCat(data)
