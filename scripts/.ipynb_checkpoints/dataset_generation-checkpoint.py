#!/usr/bin/env python
# coding: utf-8

# # FYP - Drug Design Active Learning : Dataset Generation

# A simple script to pre-process the dirty data from the 3 given datasets and combine them in an intelligent manner to be used later.

# Requirements -
# 1. PyArrow
# 2. tqdm

# In[1]:


from dask.distributed import Client, progress
import dask.dataframe as dd
import numpy as np
import os


# In[2]:


client = Client(n_workers=4, threads_per_worker=8, memory_limit='8GB')
client


# In[3]:


path_1 = os.path.dirname(os.getcwd()) + '/datasets/ci9b00375_si_001.txt'
path_tableS2 = os.path.dirname(os.getcwd()) + '/datasets/ci9b00375_si_002.txt'
path_tableS3 = os.path.dirname(os.getcwd()) + '/datasets/ci9b00375_si_003.txt'
os.path.isfile(path_1)


# In[4]:


path_1


# In[5]:


df_1 = dd.read_csv(path_1,
                   dtype= {'#cmpd':np.int64},
                   sep='\t',
                   on_bad_lines='warn')
df_1 = df_1.dropna()


# In[6]:


df_1.compute().info()


# In[7]:


df_1.compute().tail()


# In[8]:


df_table2 = dd.read_csv(path_tableS2,
                        sep=' ',
                        usecols = ['CompoundID', 'AssayID', 'expt_pIC50', 'max2‐pQSAR_pIC50', 'Clustering'],
                        dtype=str,
                        #dtype = {'AssayID':np.int64,
                        #         'expt_pIC50':np.double,
                        #         'max2‐pQSAR_pIC50':np.double
                        #        },
                        skiprows=46,
                        on_bad_lines='warn', 
                        skip_blank_lines=True)

df_table2['expt_pIC50'] = df_table2['expt_pIC50'].str.replace('\u2010', '-', regex=False).astype(np.float32)
df_table2['max2‐pQSAR_pIC50'] = df_table2['max2‐pQSAR_pIC50'].str.replace('\u2010', '-', regex=False).astype(np.float32)
df_table2 = df_table2.dropna()
df_table2['AssayID'] = df_table2['AssayID'].astype(np.int64)
df_table2 = df_table2.rename(columns = {'AssayID':'assay_id'})


# In[9]:


df_table2.compute().info()


# In[10]:


df_table2.compute().tail()


# In[11]:


df_table3 = dd.read_csv(path_tableS3,
                        sep=',',
                        #usecols = ['CompoundID', 'AssayID', 'expt_pIC50', 'max2‐pQSAR_pIC50', 'Clustering'],
                        dtype=str,
                        #dtype = {'AssayID':np.int64,
                        #         'expt_pIC50':np.double,
                        #         'max2‐pQSAR_pIC50':np.double
                        #        },
                        skiprows=47,
                        on_bad_lines='warn', 
                        skip_blank_lines=True)
df_table3 = df_table3.dropna()
df_table3['SMILES'] = df_table3['SMILES'].str.replace('\u2010', '-', regex=False)


# In[12]:


df_table3.compute().info()


# In[13]:


df_table3.compute().tail()


# In[14]:


temp_df = dd.merge(df_table2,df_table3,
                  how='left',
                  on='CompoundID',
                  )


# In[15]:


temp_df.compute().info()


# In[16]:


temp_df.tail()


# In[17]:


temp_df = dd.merge(temp_df,df_1,
                  how='left',
                  on='assay_id',
                  )
temp_df = temp_df.rename(columns = {'SMILES':'smiles'})


# In[18]:


temp_df.compute().info()


# In[19]:


temp_df.compute().tail()


# In[20]:


molecular_space = temp_df.compute()[['CompoundID','smiles']].drop_duplicates(subset='CompoundID')
molecular_space.head()


# In[21]:


assay_id_space = temp_df.compute().drop_duplicates(subset=['assay_id','#cmpd'])[['assay_id','#cmpd']]
assay_id_space.head()


# In[22]:


len(assay_id_space)


# In[23]:


target_family_space = temp_df.compute().drop_duplicates(subset=['assay_type','target_family'])[['assay_type','target_family']]
target_family_space.head()


# In[24]:


len(target_family_space)


# In[26]:


if os.path.isdir(os.path.dirname(os.getcwd())+ '/temp_data/default/') == False:
    os.mkdir(os.path.dirname(os.getcwd())+ '/temp_data/default/')
# Saving the complete preprocessed file
try:
    temp_df.compute().to_csv(os.path.dirname(os.getcwd()) +'/temp_data/default/pre_processed_file.csv')
except FileNotFoundError:
    print('Storage File ".csv" unable to be written')
    
try:
    temp_df.compute().to_parquet(os.path.dirname(os.getcwd()) + '/temp_data/default/pre_processed_file.parquet')
except FileNotFoundError:
    print('Storage File ".parquet" unable to be written')

# Saving the molecular space file
try:
    molecular_space.to_csv(os.path.dirname(os.getcwd()) + '/temp_data/default/molecular_space_file.csv')
except FileNotFoundError:
    print('Molecular Space ".csv" unable to be written')
    
try:
    molecular_space.to_parquet(os.path.dirname(os.getcwd()) + '/temp_data/default/molecular_space_file.parquet')
except FileNotFoundError:
    print('Molecular Space ".parquet" unable to be written')

# Saving the assay_id + compound_id file
try:
    assay_id_space.to_csv(os.path.dirname(os.getcwd()) + '/temp_data/default/assay_id_file.csv')
except FileNotFoundError:
    print('Assay ID list ".csv" unable to be written')
    
try:
    assay_id_space.to_parquet(os.path.dirname(os.getcwd()) + '/temp_data/default/assay_id_file.parquet')
except FileNotFoundError:
    print('Assay ID list ".parquet" unable to be written')

# Saving the assay_type + target_family file
try:
    target_family_space.to_csv(os.path.dirname(os.getcwd()) + '/temp_data/default/target_family_file.csv')
except FileNotFoundError:
    print('Target family list ".csv" unable to be written')
    
try:
    target_family_space.to_parquet(os.path.dirname(os.getcwd()) + '/temp_data/default/target_family_file.parquet')
except FileNotFoundError:
    print('Target family list ".parquet" unable to be written')


# In[ ]:




