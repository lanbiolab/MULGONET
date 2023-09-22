from keras.callbacks import LearningRateScheduler
from keras import backend as K
import random
import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from evaluates import evaluates,get_class_weight

from preprocessing import Get_Node_relationships,Get_pathway_gene_relationships,Preprocessing,gene_pathways_matrix

from MULGONET import create_model

from weight_coef import get_important_score

from Comparison import Creat_RBFSVM


# Define a learning rate update function
def myScheduler(epoch):

    if epoch % 70 == 0 and epoch != 0:
        lr = K.get_value(models.optimizer.lr)
        K.set_value(models.optimizer.lr, lr * 0.5)
    return K.get_value(models.optimizer.lr)


# Define a learning rate callback function
myReduce_lr = LearningRateScheduler(myScheduler)


# gain the pathway-pathway relations
Get_Node_relation,pathway_union_bp  = Get_Node_relationships('biological_process','./data/GO_hierarchical_structure.csv','bp')

Get_Node_relation_mf,pathway_union_mf = Get_Node_relationships('molecular_function','./data/GO_hierarchical_structure.csv','mf')

gene_data_bp = Get_pathway_gene_relationships('bp')

gene_data_mf = Get_pathway_gene_relationships('mf')



# Obtaining the dataset
meth_data, cnv_amp, cnv_del, exp_data, response = Preprocessing(Get_Node_relation,Get_Node_relation_mf,gene_data_bp,gene_data_mf)

#Class weight coefficient
x_0 , x_1 = get_class_weight(response)

gene_pathway_bp_dfss,gene_pathway_mf_dfss,gene_pathway_mf_dfs,gene_pathway_bp_dfs = gene_pathways_matrix(meth_data, cnv_amp, cnv_del, exp_data,pathway_union_bp,pathway_union_mf,gene_data_bp,gene_data_mf)



multi_data = pd.concat([meth_data,cnv_amp,cnv_del,exp_data],axis = 1)
response = response.loc[multi_data.index]



skf = StratifiedKFold(n_splits=5,shuffle=True) #,shuffle=True class_weight = {0:x_0,1:x_1},random_state=1029)

multi_data_train = multi_data.values
multi_data_test = response['response'].values

#Select positive samples
positive_example = response['response'][response['response']>0].index
positive_meth = meth_data.loc[positive_example]
positive_amp = cnv_amp.loc[positive_example]
positive_del = cnv_del.loc[positive_example]
positive_exp = exp_data.loc[positive_example]
total_positive_data = pd.concat([positive_meth, positive_amp, positive_del, positive_exp], axis=1)

total_score = []

for i in range(0,1):
    kfscore = []
    p= 0
    for train_index, test_index in skf.split(multi_data_train,multi_data_test):

        multi_data_train_x, multi_data_test_x = multi_data_train[train_index], multi_data_train[test_index] #突变数据的划分  x 自变量


        multi_data_train_y, multi_data_test_y = multi_data_test[train_index], multi_data_test[test_index]   #  y 应变量


         #,class_weight = {0:x_0,1:x_1},
        opt = keras.optimizers.Adam(lr = 0.001)

        models = create_model(multi_data_train_x,gene_pathway_bp_dfss,Get_Node_relation, gene_pathway_mf_dfss,Get_Node_relation_mf,
                 bp_net=False, mf_net=False, optimizers=opt)

        #     keras.utils.plot_model(models, show_shapes=True)   callbacks=[myReduce_lr],

        models.fit(multi_data_train_x,multi_data_train_y,
                          epochs=150,batch_size = 64,class_weight = {0:x_0,1:x_1},
                          validation_data=( multi_data_test_x,multi_data_test_y)
                  )

        y_pred = models.predict(multi_data_test_x)

        print('pre, acc, rec, f1, auc, aupr, auprc')
        print(evaluates(multi_data_test_y, y_pred))

        kfscore.append(evaluates(multi_data_test_y, y_pred))


        #Calculating the weights
        get_important_score(p, models, Get_Node_relation, Get_Node_relation_mf, gene_pathway_bp_dfs,
                            gene_pathway_mf_dfs, total_positive_data, 'weight_coef')

        p = p + 1

    #Average value
    kfscore = np.array(kfscore).sum(axis= 0)/5.0     #pre,acc,rec,auc
    print('Cross validated mean score : pre, acc, rec, f1, auc, aupr, auprc')
    print(kfscore)
    total_score.append(kfscore)

print(total_score)



# concatong_date
tol_data = pd.concat([meth_data,cnv_amp,cnv_del,exp_data],axis= 1)
multi_data_train = tol_data.values

#svm
skf = StratifiedKFold(n_splits=5, shuffle=True)
kfscore = []

for train_index, test_index in skf.split(multi_data_train, multi_data_test):
    score = list(Creat_RBFSVM(multi_data_train, multi_data_test, train_index, test_index,class_weight={0:x_0,1:x_1}))
    print('pre, acc, rec, f1, auc, aupr, auprc')
    print(score)
    kfscore.append(score)

print('Cross validated mean score : pre, acc, rec, f1, auc, aupr, auprc')
print(np.array(kfscore).sum(axis=0) / 5.0)



