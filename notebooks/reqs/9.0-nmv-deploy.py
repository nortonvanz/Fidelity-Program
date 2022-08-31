#!/usr/bin/env python
# coding: utf-8

# # Environment Preparation

# ## Imports

# In[104]:


import re
import io
import boto3
import numpy             as np
import pandas            as pd
import seaborn           as sns
import datetime          as dt
import umap.umap_        as umap

from matplotlib               import pyplot as plt
from tabulate                 import tabulate

from sklearn                  import preprocessing as pp
from sklearn                  import metrics as m
from scipy.cluster            import hierarchy as hc

from sqlalchemy               import create_engine
from sqlalchemy.pool          import NullPool
from postgre_credentials      import *


# In[105]:


from IPython.core.display     import HTML, Image
def jupyter_settings():
    """ Optimize general settings, standardize plot sizes, etc. """
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 20
    display( HTML( '<style>.container { width:100% !important; }</style>') )
    pd.set_option( 'display.expand_frame_repr', False )
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 30)
    sns.set()
jupyter_settings()


# # Data Collection

# In[9]:


#list my buckets, its files and load a file from S3 AWS:
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "loyals-dataset"

#buckets
client = boto3.client("s3", region_name=AWS_REGION)
response = client.list_buckets()

print("Listing Amazon S3 Buckets:")
for bucket in response['Buckets']:
    print(f"-- {bucket['Name']}")

#files
s3_resource = boto3.resource("s3", region_name=AWS_REGION)
s3_bucket = s3_resource.Bucket(S3_BUCKET_NAME)         
print('Listing Amazon S3 Bucket objects/files:')

for obj in s3_bucket.objects.all():
    print(f'-- {obj.key}')   
          
#load files
df_raw = pd.read_csv(io.BytesIO(obj.get()['Body'].read()), encoding='iso-8859-1') 


# In[ ]:


#read local data
# path = '/Users/home/repos/pa005_fidelity_program/'
# df_raw = pd.read_csv(path +'data/raw/Ecommerce.csv')#, encoding='unicode_escape' #encoding='iso-8859-1'


# In[10]:


df_raw.head()


# In[11]:


df_raw = df_raw.drop('Unnamed: 8', axis=1).copy()
df_raw.sample(3)


# # Data Description

# In[12]:


df1 = df_raw.copy()


# ## Rename Columns

# In[13]:


df1.sample(3)


# In[14]:


df1.columns


# In[15]:


df1.columns = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date',
       'unit_price', 'customer_id', 'country']


# ## Feature Description 

# In[18]:


# Explain feature meanings
tab_meanings = [['Columns', 'Meaning'],
        ['invoice_no', 'unique identifier of each transaction'],
        ['stock_code', 'item code'],
        ['description', 'item name'],
        ['quantity', 'quantity of each item purchased per transaction'],
        ['invoice_date', 'the day the transaction took place'],
        ['unit_price', 'product price per unit'],
        ['customer_id', 'unique customer identifier'],
        ['country', 'customer\'s country of residence']
      ]
print(tabulate(tab_meanings, headers='firstrow', stralign='left', tablefmt='simple'))


# In[19]:


df1.sample(3)


# ## Data Dimensions

# In[20]:


print(f'Number of rows: {df1.shape[0]}')
print(f'Number of columns: {df1.shape[1]}')


# In[21]:


df1.info()


# ## Check NA

# In[22]:


df1.isna().sum()


# ## Replace NA

# In[26]:


#separate dataset
df_missing = df1.loc[df1['customer_id'].isna()]
df_not_missing = df1.loc[~df1['customer_id'].isna()]

#create reference
df_invoice = pd.DataFrame(df_missing['invoice_no'].drop_duplicates() )
df_invoice['customer_id'] = np.arange(19000,19000+len(df_invoice),1)

# merge original with reference dataframe
df1 = pd.merge(df1, df_invoice, on='invoice_no', how='left')

#coalesce equivalent
df1['customer_id'] = df1['customer_id_x'].combine_first(df1['customer_id_y'])

# drop extra columns
df1 = df1.drop(columns = ['customer_id_x','customer_id_y'], axis=1 )


# In[27]:


#check NA
df1.isna().sum()


# ## Change Types

# In[28]:


#correct data types ensure correct calculations using the columns on next sessions


# In[29]:


df1.dtypes


# In[30]:


df1.sample(3)


# In[31]:


#invoice_date
df1['invoice_date'] = pd.to_datetime(df1['invoice_date'], format='%d-%b-%y')

#customer_id
df1['customer_id'] = df1['customer_id'].astype(int)


# In[32]:


df1.sample(3)


# In[33]:


df1.dtypes


# ## Descriptive Statistics

# In[34]:


#here we identify state of variables, but take action just on proper sections ahead.
num_attributes = df1.select_dtypes(include=['int64','float64'])
cat_attributes = df1.select_dtypes(exclude=['int64','float64','datetime64[ns]'])


# ### Numerical Attributes

# In[35]:


num_attributes.head()


# In[36]:


# central tendency - mean, median
ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T
             
# dispersion - desvio padrão, minimo, maximo, range, skew, kurtosis
d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T
d2 = pd.DataFrame( num_attributes.apply( np.min ) ).T
d3 = pd.DataFrame( num_attributes.apply( np.max ) ).T
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T

# concatenate
num_metrics = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()
num_metrics.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
num_metrics


# ### Cathegorical Attributes

# In[37]:


cat_attributes.head()


# #### invoice_no

# In[40]:


# problem: we have invoice_no with letters and numbers
#cat_attributes['invoice_no'].astype( int )  # -> error: ex.'C536379'

# identify number of sales with characters on invoice_no: 
df_letter_invoices = df1.loc[df1['invoice_no'].apply( lambda x: bool( re.search( '[^0-9]+', x ) ) ), :]
df_letter_invoices


# In[41]:


#looks like all invoice_no with C, has negative quantity. Lets check:
print( f'Total number of invoices: {len( df_letter_invoices )}')
print( f'Total number of negative quantity: {len( df_letter_invoices[ df_letter_invoices["quantity"] < 0 ] )}') 
#3 of difference, let's ignore them


# #### stock_code

# In[ ]:


df1['stock_code']


# In[ ]:


# get stock_codes with only letters
df1.loc[df1['stock_code'].apply( lambda x: bool( re.search( '^[a-zA-Z]+$', x ) ) ), 'stock_code'].unique()
#now we have new stock_codes!


# In[ ]:


#find samples:
df1.loc[df1['stock_code'].apply( lambda x: bool( re.search( '^[a-zA-Z]+$', x ) ) ) ].sample(10)


# # Variable Filtering

# In[42]:


df2 = df1.copy()
#df2.to_csv("../data/interim/cycle8/df2_data_description_done.csv")


# In[43]:


#remove bad users:
df2 = df2.loc[~df2['customer_id'].isin([16446, 15749, 12346]) ]

##Cat Attr

#2. stock_code - remove useless values:
df2 = df2.loc[~df2['stock_code'].isin( ['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY','DCGSSGIRL', 'PADS', 'B', 'CRUK'] ) ]
#3. description - remove useless feature:
df2 = df2.drop('description', axis=1)
#4. country - remove 2 values:
df2 = df2[~df2['country'].isin( ['European Community', 'Unspecified' ] ) ]

##Num Attr
#2. unit_price - remove <0.03:
df2 = df2.loc[df2['unit_price'] >= 0.04 ]
#1. quantity - separete into 2 datasets:
df2_returns = df2.loc[df2['quantity'] < 0]
df2_purchases = df2.loc[df2['quantity'] > 0] #there is no quantity == 0.


# # Feature Engeneering

# In[44]:


df3 = df2.copy()
df3_returns = df2_returns.copy()
df3_purchases = df2_purchases.copy()
#df3.to_csv("../data/interim/cycle8/df3_var_filtering_done.csv")
#df3_returns.to_csv("../data/interim/cycle8/df3_returns.csv")
#df3_purchases.to_csv("../data/interim/cycle8/df3_purchases.csv")


# In[45]:


#create the referente table with uniques customer_id (and reset index) 
df_ref = df3.drop(['invoice_no', 'stock_code', 'quantity', 'invoice_date',
       'unit_price', 'country'], axis=1).drop_duplicates(ignore_index=True).copy()


# In[46]:


df_ref


# ## Gross Revenue

# In[47]:


# Gross Revenue: (quantity * price of each purchase)
    #here, we just want to consider gross revenue from sales, not discounting returns, so lets use df3_purchases
df3_purchases['gross_revenue'] = df3_purchases['quantity'] * df3_purchases['unit_price']
df_monetary = df3_purchases[['customer_id','gross_revenue']].groupby('customer_id').sum().reset_index()
df_ref = pd.merge( df_ref, df_monetary, on='customer_id', how='left' )
df_ref.isna().sum()


# ## Recency

# In[48]:


# Recency: number of days since last purchase
    #here, we want to know the last day this customer bought. If he returned, we don't want to count that day as last purchase, so let's consider just df3_purchases
df_recency = df3_purchases[['customer_id','invoice_date']].groupby('customer_id').max().reset_index()
df_recency['recency_days'] = (df3_purchases['invoice_date'].max() - df_recency['invoice_date']).dt.days
df_recency = df_recency[['customer_id', 'recency_days']].copy()
df_ref = pd.merge( df_ref, df_recency, on='customer_id', how='left' )
df_ref.isna().sum()


# ## Invoice Quantity

# In[49]:


#quantity of invoices per customer
    #consider just purchases, not returns
df_invoice = df3_purchases[['customer_id','invoice_no']].drop_duplicates().groupby('customer_id').count().reset_index().rename(columns={'invoice_no':"qtt_invoices"})
df_ref = pd.merge( df_ref, df_invoice, on='customer_id', how='left')
df_ref.isna().sum()


# ## Unique Products

# In[50]:


#quantity of distinct products per customer
    #consider just purchases, not returns
df_invoice = df3_purchases[['customer_id','stock_code']].drop_duplicates().groupby('customer_id').count().reset_index().rename(columns={'stock_code':"unique_products"})
df_ref = pd.merge( df_ref, df_invoice, on='customer_id', how='left')
df_ref.isna().sum()


# ## Unique Items

# In[51]:


#quantity of items purchased per customer
    #consider just purchases, not returns
df_prod_quantity = df3_purchases[['customer_id','quantity']].groupby('customer_id').sum().reset_index().rename(columns ={'quantity':'unique_items'})
df_ref = pd.merge( df_ref, df_prod_quantity, on='customer_id', how='left')
df_ref.isna().sum()


# ## Daily Purchase Rate

# In[52]:


# purchase rate per day during the period
#per customer and invoice, get min and max invoice date, total days between min and max, and quantity of invoices
df_aux = ( df3_purchases[['customer_id', 'invoice_no', 'invoice_date']].drop_duplicates()
            .groupby( 'customer_id')
            #new column name ('apply on this columns', 'operation')
            .agg( max_invoice_date = ( 'invoice_date', 'max' ), 
                  min_invoice_date = ( 'invoice_date', 'min' ),
                  invoice_total_days= ( 'invoice_date', lambda x: ((x.max() - x.min()).days)+1),
                  invoice_count = ( 'invoice_no', 'count' ) ) ).reset_index()

# Frequency: invoice_count / invoice_total_days (if invoice_total_days != 0)
df_aux['daily_purchase_rate'] = df_aux[['invoice_count', 'invoice_total_days']].apply( 
    lambda x: x['invoice_count'] / x['invoice_total_days'] 
         if  x['invoice_total_days'] != 0 else 0, axis=1 )

# Merge
df_ref = pd.merge( df_ref, df_aux[['customer_id', 'daily_purchase_rate']], on='customer_id', how='left' )

df_ref.isna().sum()


# ## Returns

# In[53]:


#number of products (items) returned per customer
df_returns = df3_returns[['customer_id','quantity']].groupby('customer_id').sum().reset_index().rename( columns={'quantity':'total_prod_returned'} )

#convert to positive, cause we know it's a devolution, no need to be negative anymore:
df_returns['total_prod_returned'] = df_returns['total_prod_returned'] * -1

#bind
df_ref = pd.merge( df_ref, df_returns, how='left', on='customer_id' )

#since df_ref has all customers (purchases + returns), we can say these 4191 with NA in total_prod_returned are from df3_purchases, so let's assign zero to them:
df_ref.loc[df_ref['total_prod_returned'].isna(), 'total_prod_returned'] = 0

df_ref.isna().sum()


# # Data Preparation

# In[54]:


df_ref.head()


# In[55]:


df_ref.isna().sum()


# In[56]:


df_ref = df_ref.dropna() #was at EDA
df5 = df_ref.copy() #because EDA Uni and Biv were deleted
df5


# In[59]:


#test each variable to define wich is the best standardization or reescaling
mm = pp.MinMaxScaler()
ss = pp.StandardScaler()
rs = pp.RobustScaler()

df5['gross_revenue'] = mm.fit_transform(df5[['gross_revenue']])
df5['recency_days'] = mm.fit_transform(df5[['recency_days']])
df5['qtt_invoices'] = mm.fit_transform(df5[['qtt_invoices']])
df5['unique_products'] = mm.fit_transform(df5[['unique_products']])
df5['unique_items'] = mm.fit_transform(df5[['unique_items']])
df5['daily_purchase_rate'] = mm.fit_transform(df5[['daily_purchase_rate']])
df5['total_prod_returned'] = mm.fit_transform(df5[['total_prod_returned']])


# In[60]:


#after transformations:
df5.head()


# # Feature Selection

# In[62]:


df6 = df5.copy()
#df6.to_csv("../data/interim/cycle8/df6_data_prep_done.csv")


# In[63]:


#select features for variable space

#drop customer_id
X = df6.drop('customer_id', axis=1).copy()
X.head()


# In[64]:


#select features
selected_features = ['gross_revenue','recency_days','unique_products','daily_purchase_rate','total_prod_returned'] #5

X = X[selected_features].copy()
X.head()


# # EDA - Data Space Study

# In[65]:


#as soon as data on original space is not organized, let's look for a better data space with embedding
X.head()


# In[66]:


#keep just Umap embedding space (the best)


# ## UMAP

# In[67]:


#UMAP basically takes a dataset in a very high dimension and returns a new dataset with the same number of samples in a space of much lower dimension than the original dimension.
#UMAP is good for a lot of data because it's faster, but it's not as verbose in visualization.


# In[70]:


reducer = umap.UMAP( random_state=42 )
embedding = reducer.fit_transform( X )
df_umap = pd.DataFrame()
df_umap['embedding_x'] = embedding[:, 0]
df_umap['embedding_y'] = embedding[:, 1]
sns.scatterplot( x='embedding_x', y='embedding_y', data=df_umap )


# In[76]:


#evaluation, passing df_umap as dataframe

#set number of clusters
clusters_em = np.arange(2,13,1)
clusters_em

#let's use scipy instead of sklearn, being a simpler implementation.
hc_list_em = []
for k in clusters_em:
    #model definition and training
    hc_model_em = hc.linkage(df_umap, 'ward')

    #model predict
    hc_labels_em = hc.fcluster(hc_model_em, k, criterion='maxclust')#to cut dendrogram

    #model perfomance (SS)
    hc_ss_em = m.silhouette_score(df_umap, hc_labels_em, metric='euclidean')
    hc_list_em.append(hc_ss_em) 


# In[77]:


#Silhouette Score (with tree-based embedding from not scaled dataset)
plt.plot(clusters_em, hc_list_em, linestyle='--', marker='o', color='b')
plt.xlabel('K');
plt.ylabel('Silhouette Score');
plt.title('Silhouette Score x K');


# # Model Training

# In[78]:


#let's keep using Umap embedding space, where on HC with 8k, we've got an SS of 0.55 
#8Ks is also a good number of cluster for business team handle actions from them
X = df_umap
X


# In[79]:


#8 clusters, as defined
k = 8

#let's use scipy instead of sklearn, being a simpler implementation.
#model definition and training
hc_model = hc.linkage(X, 'ward')

#model predict
hc_labels = hc.fcluster(hc_model, k, criterion='maxclust')#to cut dendrogram

#model perfomance (SS)
hc_ss = m.silhouette_score(X, hc_labels, metric='euclidean') 
print(f'SS Value for {k} clusters: {hc_ss}')


# # Cluster Analysis

# ## Cluster Profile

# In[85]:


df_ref


# In[89]:


df9p = df_ref.copy() #df_ref contains original values (not reescaled), to use in cluster profile
ids_and_vars_selected = ['customer_id','gross_revenue','recency_days','unique_products','daily_purchase_rate','total_prod_returned']
df9p = df9p[ids_and_vars_selected]
df9p['cluster'] = hc_labels

#change dtypes
df9p['recency_days'] = df9p['recency_days'].astype(int)
df9p['unique_products'] = df9p['unique_products'].astype(int)
df9p['total_prod_returned'] = df9p['total_prod_returned'].astype(int)

#record timestamp of training
df9p['last_training'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
df9p


# In[91]:


#building df_cluster

# Number of customer
df_cluster = df9p[['customer_id', 'cluster']].groupby('cluster').count().reset_index()
df_cluster['perc_customer'] = df_cluster['customer_id'] / df_cluster['customer_id'].sum()*100

# Avg gross revenue
df_avg_gross_revenue = df9p[['gross_revenue','cluster']].groupby('cluster').mean().reset_index()
df_cluster = pd.merge(df_cluster, df_avg_gross_revenue, how='inner', on='cluster')

# Avg recency days
df_avg_recency_days = df9p[['recency_days','cluster']].groupby('cluster').mean().reset_index()
df_cluster = pd.merge(df_cluster, df_avg_recency_days, how='inner', on='cluster')

# Avg unique products
df_avg_unique_products = df9p[['unique_products','cluster']].groupby('cluster').mean().reset_index()
df_cluster = pd.merge(df_cluster, df_avg_unique_products, how='inner', on='cluster')

# Avg daily purchase rate
df_avg_daily_purchase_rate = df9p[['daily_purchase_rate','cluster']].groupby('cluster').mean().reset_index()
df_cluster = pd.merge(df_cluster, df_avg_daily_purchase_rate, how='inner', on='cluster')

# Avg total products returned
df_avg_total_prod_returned = df9p[['total_prod_returned','cluster']].groupby('cluster').mean().reset_index()
df_cluster = pd.merge(df_cluster, df_avg_total_prod_returned, how='inner', on='cluster')

# cluster profiles
df_cluster.sort_values('gross_revenue', ascending=False)


# In[ ]:


#df_cluster.to_csv('../data/interim/cycle8/df_cluster_kmeans_8k.csv')


# In[92]:


#generate the repport dinamically:
for i in range(len(df_cluster['customer_id'])):
    print(f""" Cluster {df_cluster['cluster'][i]}:
    -Number of customers: {df_cluster['customer_id'][i]} ({round(df_cluster['perc_customer'][i])}%)
    -Average revenue: ${round(df_cluster['gross_revenue'][i])}
    -Average recency: by each {round(df_cluster['recency_days'][i])} days ({round(df_cluster['recency_days'][i]/7)} week(s))
    -Average unique products purchased: {round(df_cluster['unique_products'][i])}  
    -Average purchases/month: {round((df_cluster['daily_purchase_rate'][i])*30 ,1)} 
    -Average total products returned: {round(df_cluster['total_prod_returned'][i])} """)


# # Deploy

# ## Insert into SQLITE

# In[93]:


df9p


# In[94]:


df9p.dtypes


# In[101]:


#create db connection (and db_file if sqlite)

#sqlite
#endpoint = 'sqlite:////Users/home/repos/pa005_fidelity_program/notebooks/loyals_db.sqlite' #local

#postgre
endpoint = f'postgresql://{pg_user}:{pg_passwd}@{pg_host}:{pg_port}'
        
db = create_engine(endpoint, poolclass=NullPool)
conn = db.connect()


# In[ ]:


# #check if table exists on sqlite
# check_table = """
#     SELECT name FROM sqlite_master WHERE type='table' AND name='loyals';
# """
# df_check = pd.read_sql_query(check_table, conn)

# #create table if does not exist
# if len(df_check) == 0:  #0 = table does not exist, 1 = table exists
#     query_create_table_loyals = """
#     CREATE TABLE loyals (
#         customer_id              INTEGER,
#         gross_revenue            REAL,
#         recency_days             INTEGER,
#         unique_products          INTEGER,
#         daily_purchase_rate      REAL,
#         total_prod_returned      INTEGER,
#         cluster                  INTEGER,
#         last_training            TEXT
#         )"""
#     conn.execute( query_create_table_loyals )
#     print('Table loyals was created!')
# else:
#     print('Table loyals exists!')


# In[ ]:


#create table
# query_create_table_loyals = """
#     CREATE TABLE loyals (
#         customer_id              INTEGER,
#         gross_revenue            REAL,
#         recency_days             INTEGER,
#         unique_products          INTEGER,
#         daily_purchase_rate      REAL,
#         total_prod_returned      INTEGER,
#         cluster                  INTEGER,
#         last_training            TEXT
#         )"""
# conn.execute( query_create_table_loyals )


# In[ ]:


# insert data into table loyals using sqlalchemy, appending data
df9p.to_sql('loyals', con=conn, if_exists='append', index=False )#index=False to ignore dataframe index


# In[102]:


#consult database
query = """
    SELECT * FROM loyals
"""
df = pd.read_sql_query(query, conn)
df


# In[103]:


conn.close() #closes connection


# In[ ]:




