#One-hot encdoing

#By pd.get_dummies
import pandas as pd 
path = ''
df = pd.read_excel(path+'feature_transformation.xlsx')
df_onehot = pd.get_dummies(df['color'],columns=['color'],prefix='it_is',prefix_sep='_')
new_df = pd.concat((df['color'],df_onehot),axis = 1)
print(new_df)
# print(pd.get_dummies(df.color))