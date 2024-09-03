import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import pandas as pd

def shap_dict(shap_values_test):
    """
    store the mean absolute shap value for each variable in a dictionary
    @param shap_values_test: the shap explainer
    @return: shap_dict
    """
    #
    if len(shap_values_test.shape)==3:
        shap_values_test.values=shap_values_test.values[:,:,1]
    shap_dic = {}
    i = 0
    while i < shap_values_test.values.shape[1]:
        sum_shap = 0
        for m in shap_values_test.values[:, i]:
            sum_shap += np.abs(m)
        shap_dic[shap_values_test.feature_names[i]] = sum_shap / shap_values_test.values.shape[0]
        i += 1
    return shap_dic

def shap_absolute_rank(shap_values_test):
    """
    calculate the rank  list in the format `[['feature_name', mean(abs(shap))]]`
    @param shap_values_test: calculated shap explainer
    @return: sorted shap rank list
    """
    # firstly,
    abs_rank_lst = []
    for i in range(shap_values_test.values.shape[1]):
        importance = sum(abs(shap_values_test.values[:, i])) / shap_values_test.values.shape[0]
        feature_name = shap_values_test.feature_names[i]
        abs_rank_lst.append([feature_name, importance])

    abs_rank_lst = sorted(abs_rank_lst, key=lambda list: list[1], reverse=True)
    return abs_rank_lst

## top 10 scatter plot
def top_10_scatter_plot(shap_values_test,var_dict,save_control=False):
    abs_rank_lst = shap_absolute_rank(shap_values_test)
    variable_to_display = [x[0] for x in abs_rank_lst if round(x[1], 2) >= 0.1]

    lim_dic = {"age": [49, 103],
               "ZincomeT": [-6.7, 5],
               "ZwealthT": [-5.5, 3],
               'Zanxiety': [-1.5, 4.5],
               'Zperceivedconstraints': [-1.5, 3.5],
               'Zconscientiousness': [-2.5, 5]}

    i = 0
    fontsize_ticks = 12
    fontsize_labels = 13
    figure, axis = plt.subplots(2, round((len(variable_to_display) + 0.1) / 2))
    figure.subplots_adjust(left=0.08, top=0.95, bottom=0.08, right=0.95)
    figure.set_figheight(8)
    figure.set_figwidth(18)

    for (m, n), subplot in np.ndenumerate(axis):
        if i < len(variable_to_display):
            var = variable_to_display[i]
            ind = shap_values_test.feature_names.index(var)
            shap_value = shap_values_test.values[:, ind]
            value = shap_values_test.data[:, ind]
            if var in lim_dic:
                lim_upper, lim_lower = lim_dic[var][1], lim_dic[var][0]
            else:
                lim_upper, lim_lower = int(value.max()) + 1, int(value.min()) - 1

            axis[m, n].scatter(value, shap_value, c=value, s=[25] * len(value), cmap='coolwarm', norm=plt.Normalize(vmin=lim_lower, vmax=lim_upper))
            axis[m, n].set_xlim(lim_lower, lim_upper)
            axis[m, n].grid(axis='y', alpha=0.4, linestyle='dashed')
            axis[m, n].axhline(y=0, color='red', linestyle='--', alpha=0.6)
            axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
            axis[m, n].spines['top'].set_visible(False)
            axis[m, n].spines['right'].set_visible(False)
            axis[m, n].set_ylabel('SHAP value', fontsize=fontsize_labels)
            axis[m, n].set_xlabel(var_dict[var], fontsize=fontsize_labels)
            axis[m, n].set_axisbelow(True)

            i += 1
    figure.tight_layout()
    plt.show()
    if save_control:
        plt.savefig(Path.cwd() / f'OX_Thesis/graphs/shap_scatter_top_10.pdf')


def shap_rank_bar_plot(shap_values_test,var_dict,max_display,save_control=False):
    """
    plot the shap rank bar graph for all variables
    @param shap_values_test: shap explainer
    @param var_dict: variable and display name dict
    @param max_display: max variable amount to display in the graph
    """
    color_blue = '#001C5B'
    # summary bar plot
    # max_display = 61
    fontsize_ticks = 20
    fontsize_labels = 21
    sum_features = 'Sum of ' + str(shap_values_test.shape[1] - max_display+1) + ' other features'
    var_dict[sum_features] = sum_features
    shap.plots.bar(shap_values_test, show=False, max_display=max_display)
    fig = plt.gcf()
    # 26，19
    fig.set_figheight(26)
    fig.set_figwidth(19)
    # fig.subplots_adjust(left=0.4, top=0.99, bottom=0.04,right=0.95)
    fig.subplots_adjust(left=0.3, top=0.99, bottom=0.04, right=0.95)

    ax = plt.gca()


    ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # xax.set_ylabel('Input Factors', fontsize=fontsize_labels)
    ax.set_xlabel('mean(|SHAP Value|)', fontsize=fontsize_labels)
    fig.tight_layout()
    if save_control:
        plt.savefig(Path.cwd() / 'graphs/mean_shap_all.pdf')

    plt.show()


def beeswarm_plot(shap_values_test,model,max_display,var_dict,save_control=False):
    #
    fontsize_ticks = 20
    fontsize_labels = 21
    shap.summary_plot(shap_values_test,
                      model.X_test,
                      show=False,
                      max_display=max_display,
                      cmap='coolwarm')
    fig = plt.gcf()
    fig.set_figheight(10)
    fig.set_figwidth(16)
    fig.subplots_adjust(left=0.28, top=0.95, right=1.01, bottom=0.1)

    ax = plt.gca()
    plt.rc('legend', fontsize=25)
    ax.set_xlabel('SHAP Value', fontsize=fontsize_labels)
    ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.set_yticklabels(ylabels)
    if save_control:
        plt.savefig(Path.cwd() / 'graphs/summary_shap.pdf')
    plt.show()


def shap_values_and_dict(model):
    explainer = shap.TreeExplainer(model.model)
    shap_values_test = explainer(pd.concat([model.X_train,model.X_test]))
    shap_dict_ = shap_dict(shap_values_test)

    return shap_values_test,shap_dict_

def shap_values_and_dict_all(model):
    explainer = shap.TreeExplainer(model.model)
    # training set
    shap_values = explainer(model.X_train)
    df_shap_data = pd.DataFrame(shap_values.data, columns=shap_values.feature_names)
    df_shap_values = pd.DataFrame(shap_values.values[:, :, 1], columns=[f'{x}_shap' for x in shap_values.feature_names])
    df_shap_data = pd.concat([df_shap_data, df_shap_values], axis=1)
    df_shap_data['dataset'] = ['train']*len(df_shap_data)

    # testing set
    shap_values  = explainer(model.X_test)
    df_shap_data_test = pd.DataFrame(shap_values.data, columns=shap_values.feature_names)
    df_shap_values_test = pd.DataFrame(shap_values.values[:, :, 1], columns=[f'{x}_shap' for x in shap_values.feature_names])
    df_shap_data_test = pd.concat([df_shap_data_test, df_shap_values_test], axis=1)
    df_shap_data_test['dataset'] = ['test'] * len(df_shap_data_test)

    # merge
    df = pd.concat([df_shap_data,df_shap_data_test],axis = 0)
    return df


def shap_overall_rank_plot(df_shaps,save_control,var_dict):

    fontsize_labels = 12

    df_shaps.sort_values(by=['SHARE'], ascending=False, inplace=True)

    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    ax.scatter(df_shaps['HRS'], df_shaps['var'], label='HRS', alpha=0.6, marker='^')
    ax.scatter(df_shaps['SHARE'], df_shaps['var'], label='SHARE', alpha=0.6, marker='1')
    ax.scatter(df_shaps['COMB'], df_shaps['var'], label='COMB', alpha=0.5, marker='d')

    ax.grid(axis='both', alpha=0.2)
    ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('mean(|SHAP Value|)', fontsize=fontsize_labels)

    ax.legend(bbox_to_anchor=(1, 1), frameon=False)
    plt.gca().invert_yaxis()
    if save_control:
        plt.savefig(Path.cwd() / f'graphs/shap_overall_ranks.pdf')

