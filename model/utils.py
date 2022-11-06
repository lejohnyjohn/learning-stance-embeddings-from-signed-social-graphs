# Author: John Pougué-Biyong
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path
import random as rd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score


# In[ ]:


class DataBuilder:
  
    def __init__(self, edge_data, dataset_name='birdwatch', save_folds=False):
        self.dataset_name = dataset_name 
        self.save_folds = save_folds
        self.topic_data_available = False
        if 'topic' in list(edge_data):
            self.topic_data_available = True
        self.__mapping(edge_data)
        self.__info()
        self.__create_subsamples()

    def __mapping(self, df):
        """ 
        Maps nodes to arbitrary indexes.
        Input:
            df pd.DataFrame: edge data (source, target, weight)
        """
        print('Mapping instances to idx...')
        self.node2idx = {}
        self.idx2node = {}
        idx = 0
        nodes = list(df.source.unique()) + list(df.target.unique())
        nodes = sorted(list(set(nodes)))
        for node in nodes:
            self.node2idx[node] = idx
            self.idx2node[idx] = node
            idx += 1
        df['source_idx'] = df['source'].apply(lambda x: self.node2idx[x])
        df['target_idx'] = df['target'].apply(lambda x: self.node2idx[x])
        if self.topic_data_available:
            self.topic2idx = {}
            self.idx2topic = {}
            idx = 0
            topics = sorted(list(df.topic.unique()))
            for topic in topics:
                self.topic2idx[topic] = idx
                self.idx2topic[idx] = topic
                idx += 1
            df['topic_idx'] = df['topic'].apply(lambda x: self.topic2idx[x])
        else:
            self.topic2idx = {'unk': 0}
            self.idx2topic = {0: 'unk'}
            df['topic'] = 'unk'
            df['topic_idx'] = 0
        edges_hetero = df         .groupby(['source_idx', 'target_idx', 'topic_idx'])         .agg({'rating': 'sum'})         .reset_index()
        edges_hetero = edges_hetero[edges_hetero.rating != 0]
        edges_hetero['weight'] = edges_hetero['rating'].apply(lambda x: 1 if x > 0 else -1)
        self.edge_data_hetero = edges_hetero
        self.size_graph = len(self.node2idx)
        print('Mapped instances.')
        topics = [key for key in dataloader.node2idx]
        values = [dataloader.node2idx[key] for key in dataloader.node2idx]
        dataframe = pd.DataFrame({'node': topics, 'node_idx': values})
        dataframe.to_csv('cached_data/CVfolds/birdwatch/node_mapping.csv', index=False)
  
    def __info(self):
        print('#nodes:', len(self.node2idx))
        print('---heterogeneous graph----')
        print('#edges:', len(self.edge_data_hetero))
        print('%edges+:', 
              str(round(len(self.edge_data_hetero[self.edge_data_hetero.weight == 1]) * 100 \
                  / len(self.edge_data_hetero))) + '%')
        if self.topic_data_available:
            print('#topics:', len(self.topic2idx))
        else:
            print('No topic data available.')
          
    def __create_subsamples(self):
        """ 
        Creates 5-CV folds.
        """ 
        if self.save_folds:
            self.training_data = {'hetero': {}}
            self.test_data = {'hetero': {}}
            print('Creating 5-CV folds...')
            X = self.edge_data_hetero[['source_idx', 'target_idx', 'topic_idx']].to_numpy()
            y = self.edge_data_hetero[['weight']].to_numpy()
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            count = 1
            for train_index, test_index in skf.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                df_train = pd.DataFrame(X_train, 
                                      columns = ['source_idx','target_idx', 'topic_idx'])
                training_nodes = set(list(df_train.source_idx.unique())                               + list(df_train.target_idx.unique())
                              )
                training_topics = set(list(df_train.topic_idx.unique()))
                X_test_unfiltered, y_test_unfiltered = X[test_index], y[test_index]
                df_test = pd.DataFrame(X_test_unfiltered, 
                                     columns = ['source_idx','target_idx','topic_idx'])
                df_test['ID'] = list(range(len(df_test)))
                df_test = df_test[(df_test.source_idx.isin(training_nodes))                                 & (df_test.target_idx.isin(training_nodes))                                 & (df_test.topic_idx.isin(training_topics))]
                X_test = X_test_unfiltered[list(df_test.ID), :]
                y_test = y_test_unfiltered[list(df_test.ID), :]
                self.training_data['hetero'][count] = pd.DataFrame(X_train, 
                                                       columns = ['source_idx','target_idx','topic_idx'])
                self.training_data['hetero'][count]['weight'] = y_train
                self.training_data['hetero'][count].to_csv('cached_data/CVfolds/{dataset}/train_hetero_{c}.csv'                 .format(dataset=self.dataset_name, c=count), index=False)
                self.test_data['hetero'][count] = pd.DataFrame(X_test, 
                                                   columns = ['source_idx','target_idx','topic_idx'])
                self.test_data['hetero'][count]['weight'] = y_test
                self.test_data['hetero'][count].to_csv('cached_data/CVfolds/{dataset}/test_hetero_{c}.csv'                 .format(dataset=self.dataset_name, c=count), index=False)
                count += 1
              
            print('Created 5-CV folds.')
          
        else:
            self.training_data = {'hetero': {}}
            self.test_data = {'hetero': {}}
            for count in range(5):
                self.training_data['hetero'][count + 1] =                 pd.read_csv('cached_data/CVfolds/{dataset}/train_hetero_{c}.csv'                 .format(dataset=self.dataset_name, c=count+1), index_col=False)
                self.test_data['hetero'][count + 1] =                 pd.read_csv('cached_data/CVfolds/{dataset}/test_hetero_{c}.csv'                 .format(dataset=self.dataset_name, c=count+1), index_col=False)


# In[ ]:


class Dataloader:
  
    def __init__(self, dataset_name='birdwatch'):
        self.dataset_name = dataset_name 
      
        print('Collecting mappings...')
        node_path = 'cached_data/CVfolds/{}/node_mapping.csv'.format(self.dataset_name)
        self.node_mapping = pd.read_csv(node_path, index_col=False)
        topic_path = 'cached_data/CVfolds/{}/topic_mapping.csv'.format(self.dataset_name)
        self.topic_mapping = pd.read_csv(topic_path, index_col=False)
        print('Collected mapping.')
      
        print('Creating mapping dicts...')
        self.idx2node = {item.node_idx: item.node for _, item in self.node_mapping.iterrows()}
        self.node2idx = {item.node: item.node_idx for _, item in self.node_mapping.iterrows()}
        self.idx2topic = {item.topic_idx: item.topic for _, item in self.topic_mapping.iterrows()}
        self.topic2idx = {item.topic: item.topic_idx for _, item in self.topic_mapping.iterrows()}
        print('Created mapping dicts.')
      
        self.size_graph = len(self.node2idx)
        self.training_data = {'hetero': {}, }
        self.test_data = {'hetero': {}, }
      
        print('\nCollecting CVfolds...')
        for count in range(5):
            path = 'cached_data/CVfolds/{dataset}/train_hetero_{c}.csv'.format(dataset=self.dataset_name, c=count+1)
            self.training_data['hetero'][count + 1] = pd.read_csv(path, index_col=False)
          
            path = 'cached_data/CVfolds/{dataset}/test_hetero_{c}.csv'.format(dataset=self.dataset_name, c=count+1)
            self.test_data['hetero'][count + 1] = pd.read_csv(path, index_col=False)

        print('Collected CVfolds.\n')
      
        if len(set(self.training_data['hetero'][1].topic_idx.tolist())) > 1:
            self.topic_data_available = True

