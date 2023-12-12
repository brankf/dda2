# -*- coding: utf-8 -*-
"""DDA2 workshop

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',30)
pd.set_option('display.max_rows',200)


#for tutorial data
# %cd /content/drive/MyDrive/Cereal_Dataset/
!ls

"""# **week 1 workshop**

*   control + enter -> auto-run the code in the cell
*   control + M + B -> add new code
"""

#to read the cereal data file into a dataframe

df = pd.read_csv('cereal.csv')

"""**QUESTION 1**"""

#unique() is like distinct, filters out the distinct data
#nunique() counts the number of unique values

df['protein'].unique()

df['protein'].nunique()

#describe() prints out the statistics of a column
#50% is the median
#only describe() will show the max of the dataset

df['protein'].describe()

df[~ df['mfr'].isin(['A', 'P'])].groupby('mfr').agg(median_carbo=('carbo', 'median')).sort_values('median_carbo', ascending=False)

#value_counts() count the number of occurence of each unique value

df['protein'].value_counts()

df[df['rating'].between(70,100)].groupby('mfr').agg(\
   no_of_prod=('name', 'nunique'), \
   max_protein=('protein', 'max'), \
   min_vitamins=('vitamins', 'min') \
).sort_values('max_protein', ascending=False)

"""**QUESTION 2**"""

df.loc[df['calories']==50, ['name', 'calories']]

#to filter out the column with a value criteria, but display all other columns as well
df.loc[df['protein']>3]

#to filter out the column with a value criteria, then displaying only the specific column
#2nd parameter is to display the specific column(s) only

#METHOD 1

df.loc[df['protein']>3, ['name', 'protein']]

#to filter out the column with a value criteria, then displaying only the specific column
#2nd parameter is to display the specific column(s) only

#METHOD 2

df.loc[df['protein'].between(4,6), ['name', 'protein']]

#to use an inclusive parameter
#to display all values except both limits in between clause
df.loc[df['protein'].between(4,6, inclusive='neither'), ['name', 'protein']]

df.loc[(df['protein']>=4) & (df['protein']<=6), ['name', 'protein']]

"""**QUESTION 3**"""

#to find out the max values of both columns
df[['sugars', 'sodium']].describe()

df.loc[(df['sugars']==15) | (df['sodium']==320), ['name', 'sugars', 'sodium']]

df.loc[(df['sugars']==15) | (df['sodium']==320), ['name', 'sugars', 'sodium']].sort_values(['sugars', 'sodium'])

df.loc[(df['sugars']==15) | (df['sodium']==320), ['name', 'sugars', 'sodium']].sort_values(['sugars', 'sodium'], ascending=False)

df['mfr'].describe()

#finding out data type of manufacturer
df['mfr'].dtypes

df['protein'].dtypes

#finding out datatype for all columns in dataframe
df.dtypes

df['mfr'].unique()

df['mfr'].nunique()

df['mfr'].value_counts()

"""# **Week 2 workshop**

**QUESTION 5**
"""

df.loc[ df['name'].str.contains('fruit')  ,  ['mfr','name']]

df.loc[ df['name'].str.lower().str.contains('fruit')  ,  ['mfr','name']]

df.loc[ df['name'].str.upper().str.contains('FRUIT')  ,  ['mfr','name']]

#for startswith(), must use lower() or upper()
df.loc[ df['name'].str.lower().str.startswith('fruit')  ,  ['mfr','name']]

df.loc[ df['name'].str.lower().str.endswith('bran')  ,  ['mfr','name']]

df.loc[ df['name'].str.contains('&')  ,  ['mfr','name']]

#does not actually edit the strings in the original data, only for this program
df['name'].str.replace('&', 'and')

"""**QUESTION 6**

- comparison between describe() and value_counts()
"""

df['mfr'].describe()

df['mfr'].value_counts()

df.loc[   (df['mfr']=='K')   |   (df['mfr']=='G')  ,   ['mfr', 'name'] ]

#sorting the mfr column alphabetically
df.loc[   (df['mfr']=='K')   |   (df['mfr']=='G')  ,   ['mfr', 'name'] ].sort_values('mfr')

#more efficient method to than the previous code to sort the column alphabetically
#to search for the top  2 most efficient manufacturers, and then sorting them
df.loc[   df['mfr'].isin(['K', 'G'])  ,   ['mfr', 'name'] ].sort_values('mfr')

#to search for the other manufacturers, excluding 'K' and 'G'
#longer way:
df.loc[   df['mfr'].isin(['A', 'N', 'P', 'Q', 'R'])  ,   ['mfr', 'name'] ].sort_values('mfr')

#to search for the other manufacturers, excluding 'K' and 'G'
# ~ can be used with other functions, not necessarily have to be .isin()
#shorter way:
df.loc[ ~ df['mfr'].isin(['K', 'G'])  ,   ['mfr', 'name'] ].sort_values('mfr')

"""**QUESTION 7**"""

#grouping all the separate mfr together into rows
#isolate the 'rating'in a list
# median() to find the median of each group of mfr
df.groupby('mfr')['rating'].median()

#sorting a series (single column with no name - ratings), not a dataframe
#a dataframe is >= 1 columns with name(s)
#if we're sorting a dataframe, must specify the column to sort
df.groupby('mfr')['rating'].median().sort_values(ascending=False)

#create a new name for a column; in this case is media_rating
#prints out a dataframe this time, not a series
#1st parameter - column to be aggregated, 2nd parameter - aggregate function
df.groupby('mfr').agg(median_rating=('rating', 'median')).sort_values('median_rating', ascending=False)

df.groupby('mfr').agg( \
    median_rating=('rating', 'median') \
)

"""**QUESTION 8**"""

#finding the median of multiple groups
#no need to put a , on the last line
#sorting by median_rating (must specify which column to sort!!!!!)
#other agg functions: 'mean', 'max', 'min', 'median', 'sum', 'std', 'nunique'
df.groupby('mfr').agg( \
    median_rating=('rating', 'median'), \
    median_sodium=('sodium', 'median'), \
    median_sugar=('sugars', 'median'), \
    median_protein=('protein', 'median'), \
    median_potass=('potass', 'median') \
).sort_values('median_rating', ascending=False)

df.groupby('mfr').agg( \
    median_rating=('rating', 'median'), \
    median_sodium=('sodium', 'median'), \
    median_sugar=('sugars', 'median'), \
    median_protein=('protein', 'median'), \
    median_potass=('potass', 'median'), \
    no_of_products=('name', 'nunique') \
).sort_values('median_rating', ascending=False)

"""# **week 3 workshop**

**QUESTION 9**
"""

df['shelf'].unique()

df['shelf'].value_counts()

"""**QUESTION 10**"""

df['type'].describe()

#consider all 77 rows, then group them in terms of types of cereals
#look for the column 'name'
df.groupby('type')['name'].nunique()

#first phrase after .agg() is the name of the new column
#followed by which column to go to, and which agg function to use
df.groupby('type').agg(no_of_prod=('name', 'nunique'))

"""**QUESTION 11**"""

#grouping by shelf, then group by manufacturer
#followed by count the number of unique occurences per product per shelf
df.groupby(['shelf', 'mfr'])['name'].nunique()

#for the printed result, 'shelf', and 'mrf' are indexes, while 'no_of_prod' is a column name
df.groupby(['shelf', 'mfr']).agg(no_of_prod=('name', 'nunique'))

#to print out 'shelf' and 'mfr' as columns too, and not as indexes -> use as_index=False
df.groupby(['shelf', 'mfr'], as_index=False).agg(no_of_prod=('name', 'nunique'))

"""**QUESTION 12**"""

df['sugars'].describe()

#to search for 75th percentile, top 25% of sugar level
df['sugars'].quantile(0.75)

#to search for 85th percentile, top 15% of sugar level
df['sugars'].quantile(0.85)

df2 = df.loc[df['sugars'] > 11]
df2

df2 = df.loc[df['sugars'] > 11, :]
df2

df2.groupby(['shelf', 'sugars']).agg(high_sugar_prod=('name', 'nunique'))

"""**EXTRA PRACTICE**"""

#converting sodium and potassium from mg to g
df['sodium_g'] = df['sodium']/ 1000
df['potass_g'] = df['potass']/ 1000

#how to have a column that stores total nutrients of a cereal
df['total_nutrients'] = df['sodium_g'] + df['potass_g'] + df['protein'] + df['fiber'] + df['vitamins'] + df['fat'] + df['sugars'] + df['carbo']

#how to calculate nutrient density
df['nutrient_density'] = df['total_nutrients'] / df['calories']

#assign importance to various nutrients

df['protein_w'] = df['protein'] * 0.3
df['fiber_w'] = df['fiber'] * 0.3
df['vitamins_w'] = df['vitamins'] * 0.3

df['sodium_w'] = df['sodium'] * 0.1
df['sugars_w'] = df['sugars'] * 0.1

"""# **WEEK 4 WORKSHOP**
- only theory part is tested as mcq, coding not tested for CT


"""

!pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

df_iris.describe()

df_iris.isnull().sum()

#data-splitting
#X for features, y for target
X = df_iris.drop('target', axis=1)
y = df_iris['target']
#4 variables for 4 outputs
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

#algorithm selection & model training
tree_model = DecisionTreeClassifier(random_state=123, min_samples_leaf=15)
tree_model.fit(X_train, y_train)

#testing of trained model
y_pred_tree = tree_model.predict(X_test)
y_pred_tree

accuracy_tree = accuracy_score(y_test, y_pred_tree)
accuracy_tree

#visualisation of the tree that we trained
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

plot_tree(tree_model)
plt.show()

"""improving tree -> show feature name"""

plot_tree(tree_model, feature_names=iris.feature_names)
plt.show()

"""improving tree -> show the class names"""

plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

"""improving tree -> hide the gini value"""

plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, impurity=False)
plt.show()

"""improving tree -> give a colour for each class"""

plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, impurity=False, filled=True)
plt.show()