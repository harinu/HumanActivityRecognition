#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print('Train Data', train.shape,'\n', train.columns)
print('\nTest Data', test.shape)


# In[3]:


print('Train labels', train['Activity'].unique(), '\nTest Labels', test['Activity'].unique())


# In[4]:


pd.crosstab(train.subject, train.Activity)


# In[ ]:


sub15 = train.loc[train['subject']==15]


# In[ ]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sb.stripplot(x='Activity', y=sub15.iloc[:,0], data=sub15, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sb.stripplot(x='Activity', y=sub15.iloc[:,1], data=sub15, jitter=True)
plt.show()


# In[ ]:


sb.clustermap(sub15.iloc[:,[0,1,2]], col_cluster=False)


# In[ ]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sb.stripplot(x='Activity', y='tBodyAcc-max()-X', data=sub15, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sb.stripplot(x='Activity', y='tBodyAcc-max()-Y', data=sub15, jitter=True)
plt.show()


# In[ ]:


sb.clustermap(sub15[['tBodyAcc-max()-X', 'tBodyAcc-max()-Y', 'tBodyAcc-max()-Z']], col_cluster=False)


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6).fit(sub15.iloc[:,:-2])
clust = pd.crosstab(kmeans.labels_, sub15['Activity'])
clust


# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


plt.plot(kmeans.cluster_centers_[np.asscalar(clust[clust.WALKING_DOWNSTAIRS!=0].index),:100], "o")


# In[ ]:


print(sub15.columns[[40, 49, 50, 51]])


# In[ ]:


import tensorflow as tf
import numpy.core.multiarray 

# load train and test data
num_labels = 6
train_x = np.asarray(train.iloc[:,:-2])
train_y = np.asarray(train.iloc[:,562])
act = np.unique(train_y)
for i in np.arange(num_labels):
    np.put(train_y, np.where(train_y==act[i]), i)
train_y = np.eye(num_labels)[train_y.astype('int')] # one-hot encoding

test_x = np.asarray(test.iloc[:,:-2])
test_y = np.asarray(test.iloc[:,562])
for i in np.arange(num_labels):
    np.put(test_y, np.where(test_y==act[i]), i)
test_y = np.eye(num_labels)[test_y.astype('int')]

# shuffle the data
seed = 456
np.random.seed(seed)
np.random.shuffle(train_x)
np.random.seed(seed)
np.random.shuffle(train_y)
np.random.seed(seed)
np.random.shuffle(test_x)
np.random.seed(seed)
np.random.shuffle(test_y)


# In[ ]:


# place holder variable for x with number of features - 561
x = tf.placeholder('float', [None, 561], name='x')
# place holder variable for y with the number of activities - 6
y = tf.placeholder('float', [None, 6], name='y')
# softmax model
def train_softmax(x):
    W = tf.Variable(tf.zeros([561, 6]), name='weights')
    b = tf.Variable(tf.zeros([6]), name='bias')
    lr = 0.25
    prediction = tf.nn.softmax(tf.matmul(x, W) + b, name='op_predict')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for epoch in range(1000):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of', 1000, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')
    
    print('Train set Accuracy:', sess.run(accuracy, feed_dict = {x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(accuracy, feed_dict = {x: test_x, y: test_y}))


# In[ ]:


train_softmax(x)


# In[ ]:


n_nodes_input = 561 # number of input features
n_nodes_hl = 30     # number of units in hidden layer
n_classes = 6       # number of activities
x = tf.placeholder('float', [None, 561])
y = tf.placeholder('float')


# In[2]:


def neural_network_model(data):
    # define weights and biases for all each layer
    hidden_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_input, n_nodes_hl], stddev=0.3)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl]))}
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl, n_classes], stddev=0.3)),
                    'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))}
    # feed forward and activations
    l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for epoch in range(1000):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of', 1000, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')
    
    print('Train set Accuracy:', sess.run(accuracy, feed_dict = {x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(accuracy, feed_dict = {x: test_x, y: test_y}))


# In[3]:


train_neural_network(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




