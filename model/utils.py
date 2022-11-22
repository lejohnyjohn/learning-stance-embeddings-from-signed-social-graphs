#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# author: John N. Pougué-Biyong, jpougue@gmail.com


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
  
    def __init__(self, 
                 edge_data, 
                 dataset_name='BirdwatchSG', 
                 save_folds=False, 
                 use_topics=True): 
        """
        Input:
            edge_data pd.DataFrame: edge data (source_idx, target_idx, topic, weight)
            dataset_name str: BirdwatchSG or TwitterSG, or your own dataset name
            save_folds bool: set to False if CVfolds for dataset_name (& use_topics settings) 
                             are already cached, True otherwise
            use_topics bool: set to False if edge-attribute information should be ignored
        """
        self.dataset_name = dataset_name 
        self.save_folds = save_folds
        use_topics = use_topics
        if use_topics:
            assert 'topic' in list(edge_data)
            self.topic_data_available = True
        else:
            self.topic_data_available = False
        self.__mapping(edge_data)
        if self.save_folds:
            self.__info()
        self.__create_subsamples()

    def __mapping(self, df):
        """ 
        Maps nodes to arbitrary indexes.
        Input:
            df pd.DataFrame: edge data
        """
        if self.save_folds:
            print('Mapping instances to idx...')
            self.node2idx = {}
            self.idx2node = {}
            idx = 0
            nodes = list(df.source_idx.unique()) + list(df.target_idx.unique())
            nodes = sorted(list(set(nodes)))
            for node in nodes:
                self.node2idx[node] = idx
                self.idx2node[idx] = node
                idx += 1
            df['source_idx'] = df['source_idx'].apply(lambda x: self.node2idx[x])
            df['target_idx'] = df['target_idx'].apply(lambda x: self.node2idx[x])
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
            edges = df             .groupby(['source_idx', 'target_idx', 'topic_idx'])             .agg({'rating': 'sum'})             .reset_index()
            edges = edges[edges.rating != 0]
            edges['weight'] = edges['rating'].apply(lambda x: 1 if x > 0 else -1)
            self.edge_data = edges
            self.size_graph = len(self.node2idx)
            print('Mapped instances.')
        
            keys = [key for key in self.node2idx]
            values = [self.node2idx[key] for key in self.node2idx]
            dataframe = pd.DataFrame({'node': keys, 'node_idx': values})
            dataframe.to_csv('../datasets/cache/{}/node_mapping.csv'.format(self.dataset_name), 
                             index=False)
        
            keys = [key for key in self.topic2idx]
            values = [self.topic2idx[key] for key in self.topic2idx]
            dataframe = pd.DataFrame({'topic': keys, 'topic_idx': values})
            dataframe.to_csv('../datasets/cache/{}/topic_mapping.csv'.format(self.dataset_name), 
                             index=False)
  
    def __info(self):
        print('--- graph----')
        print('#nodes:', len(self.node2idx))
        print('#edges:', len(self.edge_data))
        print('%edges+:', 
              str(round(len(self.edge_data[self.edge_data.weight == 1]) * 100 \
                  / len(self.edge_data))) + '%')
        if self.topic_data_available:
            print('#topics:', len(self.topic2idx))
        else:
            print('No topic used.')
          
    def __create_subsamples(self):
        """ 
        Creates 5-CV folds.
        """ 
        if self.save_folds:
            self.training_data = {}
            self.test_data = {}
            print('Creating 5-CV folds...')
            X = self.edge_data[['source_idx', 'target_idx', 'topic_idx']].to_numpy()
            y = self.edge_data[['weight']].to_numpy()
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
                self.training_data[count] = pd.DataFrame(X_train, 
                                                         columns = ['source_idx','target_idx','topic_idx'])
                self.training_data[count]['weight'] = y_train
                self.training_data[count].to_csv('../datasets/cache/{dataset}/CVfolds/train_{c}.csv'                         .format(dataset=self.dataset_name, c=count), index=False)
                self.test_data[count] = pd.DataFrame(X_test, 
                                                     columns = ['source_idx','target_idx','topic_idx'])
                self.test_data[count]['weight'] = y_test
                self.test_data[count].to_csv('../datasets/cache/{dataset}/CVfolds/test_{c}.csv'                         .format(dataset=self.dataset_name, c=count), index=False)
                count += 1
            print('Created 5-CV folds.')
        else:
            self.training_data = {}
            self.test_data = {}
            for count in range(5):
                self.training_data[count + 1] =                     pd.read_csv('../datasets/cache/{dataset}/CVfolds/train_{c}.csv'                         .format(dataset=self.dataset_name, c=count+1), index_col=False)
                self.test_data[count + 1] =                     pd.read_csv('../datasets/cache/{dataset}/CVfolds/test_{c}.csv'                         .format(dataset=self.dataset_name, c=count+1), index_col=False)


# In[ ]:


class Dataloader:
  
    def __init__(self, dataset_name='BirdwatchSG'):
        self.dataset_name = dataset_name 
      
        print('Collecting mappings...')
        node_path = '../datasets/cache/{}/node_mapping.csv'.format(self.dataset_name)
        self.node_mapping = pd.read_csv(node_path, index_col=False)
        topic_path = '../datasets/cache/{}/topic_mapping.csv'.format(self.dataset_name)
        self.topic_mapping = pd.read_csv(topic_path, index_col=False)
        print('Collected mapping.')
      
        print('Creating mapping dicts...')
        self.idx2node = {item.node_idx: item.node for _, item in self.node_mapping.iterrows()}
        self.node2idx = {item.node: item.node_idx for _, item in self.node_mapping.iterrows()}
        self.idx2topic = {item.topic_idx: item.topic for _, item in self.topic_mapping.iterrows()}
        self.topic2idx = {item.topic: item.topic_idx for _, item in self.topic_mapping.iterrows()}
        print('Created mapping dicts.')
      
        self.size_graph = len(self.node2idx)
        self.training_data = {}
        self.test_data = {}
      
        print('\nCollecting CVfolds...')
        for count in range(5):
            path = '../datasets/cache/{dataset}/CVfolds/train_{c}.csv'.format(dataset=self.dataset_name, c=count+1)
            self.training_data[count + 1] = pd.read_csv(path, index_col=False)
          
            path = '../datasets/cache/{dataset}/CVfolds/test_{c}.csv'.format(dataset=self.dataset_name, c=count+1)
            self.test_data[count + 1] = pd.read_csv(path, index_col=False)

        print('Collected CVfolds.\n')
      
        if len(set(self.training_data[1].topic_idx.tolist())) > 1:
            self.topic_data_available = True

