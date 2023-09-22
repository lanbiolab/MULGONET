

from single_inputs_IntegratedGradients.model_utils import get_layers, get_coef_importance

import pandas as pd
import numpy as np


def get_coef_importances(model, X_train, y_train, target=-1, feature_importance='deepexplain_grad*input'):
    coef_ = get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=False)
    return coef_




def get_important_score(item_times, models,Get_Node_relation,Get_Node_relation_mf,gene_pathway_bp_dfs,gene_pathway_mf_dfs,total_positive_data,coef_path):
    item_times = 'h' + str(item_times)
    coef_ = get_coef_importances(models, total_positive_data.values, np.ones(total_positive_data.shape[0]), target=-1,
                                 feature_importance='deepexplain_deeplift')

    inputs = list(total_positive_data.columns)

    inputs = ['input_multi']
    bp_net = ['h0_bp', 'h1_bp', 'h2_bp', 'h3_bp', 'h4_bp']
    mf_net = ['h0_mf', 'h1_mf', 'h2_mf', 'h3_mf', 'h4_mf']

    data_type_id = []
    data_type = ['meth', 'amp', 'del', 'exp']
    for i in range(0, len(gene_pathway_mf_dfs)):
        for j in gene_pathway_bp_dfs[i]:
            data_type_id.append(j + "_" + data_type[i])

    inputs_key = data_type_id
    net_key = {'h0_bp': Get_Node_relation[3].index, 'h1_bp': Get_Node_relation[3].columns,
               'h2_bp': Get_Node_relation[2].columns, 'h3_bp': Get_Node_relation[1].columns,
               'h4_bp': Get_Node_relation[0].columns, 'h0_mf': Get_Node_relation_mf[3].index,
               'h1_mf': Get_Node_relation_mf[3].columns, 'h2_mf': Get_Node_relation_mf[2].columns,
               'h3_mf': Get_Node_relation_mf[1].columns, 'h4_mf': Get_Node_relation_mf[0].columns}

    kesys = list(coef_.keys())

    for i in kesys:

        if 'input' in i:
            values = coef_[inputs[0]]
            key = list(inputs_key)
            temp_df = pd.DataFrame(index=key)
            temp_df['values'] = values
            #             temp_df = temp_df.sort_values('values',ascending=False)
            temp_df.to_csv('./{}\\{}\\{}.csv'.format(coef_path,item_times, i),
                           encoding='utf-8')

        else:
            key = net_key[i]
            values = coef_[i]
            temp_df = pd.DataFrame(index=key)
            temp_df['values'] = values
            temp_df = temp_df.sort_values('values', ascending=False)
            temp_df.to_csv('./{}\\{}\\{}.csv'.format(coef_path,item_times, i),
                           encoding='utf-8')

