import time
import random
from src import params
import pandas as pd
from src import Models
from pathlib import Path
from src.Evaluate import metric
from src import Shap
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')


model_params = params.model_params
colors = {'blue':'#1f77b4','orange':'#ff7f0e','green':'#2ca02c'}

def seed_analysis():

    domain_name = 'combination_all'
    df,model_params['domain_dict'][domain_name] = params.read_merged_data()
    model_params['var_dict']['dataset']='Datasource'

    df_seed_selection_lst=pd.DataFrame(columns=['seed','model','imv','roc-auc','pr-auc','f1','efron_r2','ffc_r2','IP'])
    df_seed_shap_performance_recorder = pd.DataFrame(columns = ['seed','model']+model_params['domain_dict'][domain_name])

    count = 0
    while count < 10000:
        seed = random.randint(1, 1000000000)
        model_selection = 'lgb'

        model_params['random_state'] = seed
        model = Models.Model_fixed_test_size(data=df, model_params=model_params, domain=domain_name, model='lgb', train_subset_size=1, order=0)

        evas = metric(model)

        temp = pd.DataFrame({'seed': seed, 'model': model_selection,
                             'imv': evas.imv, 'roc-auc': evas.auc_score,
                             'pr-auc': evas.pr_auc, 'f1': evas.pr_f1,
                             'efron_r2': evas.efron_rsquare, 'ffc_r2': evas.ffc_r2, 'IP': evas.pr_no_skill}, index=[0])
        df_seed_selection_lst.loc[len(df_seed_selection_lst),] = temp.loc[0,]

        # shap zone
        shap_values_test, shap_dict = Shap.shap_values_and_dict(model)

        # store the mean absolute shap value for each variable in a dictionary
        if len(shap_values_test.shape) == 3:
            shap_values_test.values = shap_values_test.values[:, :, 1]
        shap_dict = {}
        i = 0
        while i < shap_values_test.values.shape[1]:
            sum_shap = 0
            for m in shap_values_test.values[:, i]:
                sum_shap += np.abs(m)
            shap_dict[shap_values_test.feature_names[i]] = sum_shap / shap_values_test.values.shape[0]
            i += 1
        shap_dict['seed'] = seed
        shap_dict['model'] = model_selection
        temp_shap = pd.DataFrame(shap_dict, index=[0])
        df_seed_shap_performance_recorder.loc[len(df_seed_shap_performance_recorder),] = temp_shap.loc[0,]

        del model, evas
        # rest zone
        if count % 100 == 0:
            print(f'now seed is {seed} and we take 10s rest')
            print(f'\n{model_selection} and seed is {seed}, count={count}')
            df_seed_selection_lst.to_csv(Path.cwd() / 'results/10000seed_comb_model_performance.csv', index=False)
            df_seed_shap_performance_recorder.to_csv(Path.cwd() / 'results/10000seed_comb_shap_values.csv', index=False)

            time.sleep(10)
        count += 1

    df_seed_selection_lst.to_csv(Path.cwd() / 'results/10000seed_comb_model_performance.csv', index=False)
    df_seed_shap_performance_recorder.to_csv(Path.cwd() / 'results/10000seed_comb_shap_values.csv', index=False)


def draw_brace(ax, mean, y, text):
    brace_width = 0.5
    text_pos_y = 1.08 * y + brace_width

    # Horizontal part (curly brace)
    ax.annotate('', xy=(mean, y), xycoords='data',
                xytext=(mean, text_pos_y), textcoords='data',
                arrowprops=dict(arrowstyle=f']-[, widthB=12, lengthB=0.5,angleB=0,widthA=0,lengthA=0', ),
                annotation_clip=False)

    # Text with expectation and variance
    ax.text(mean, text_pos_y, text, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='Round', fc='white'))

def seed_plot(df_eval):

    fig, ax = plt.subplots(2, 2)
    plt.rcParams["figure.figsize"] = [12, 8]
    count = 0
    colums = ['pr-auc', 'imv', 'efron_r2', 'ffc_r2']
    column_dict = {'roc-auc': 'ROC-AUC Score', 'pr-auc': 'PR-AUC Score', 'f1': 'F1', 'efron_r2': 'Efron R2', 'ffc_r2': 'FFC R2', 'imv': 'IMV'}
    fig.subplots_adjust(left=0.09, top=0.98, bottom=0.06, right=0.95)

    # colors = ['#001c54', '#E89818']

    letter_fontsize = 15
    label_fontsize = 13
    for (m, n), subplot in np.ndenumerate(ax):
        sns.distplot(df_eval[colums[count]],
                     hist_kws={'facecolor': colors['blue'], 'edgecolor': 'k', 'alpha': 0.9 },
                     kde_kws={'color': colors['orange']}, ax=ax[m, n], bins=20)
        # ax[m,n].hist(df_eval[colums[count]],color=color_blue,alpha=0.75,bins=30,edgecolor='black')
        ax[m, n].set_xlabel(column_dict[colums[count]], fontsize=label_fontsize + 1, weight='bold')
        ax[m, n].set_ylabel('Density', fontsize=label_fontsize + 1)
        ax[m, n].grid(alpha=0.4)
        ax[m, n].tick_params(axis='both', which='major', labelsize=label_fontsize)

        # annotation part

        stats_text = f"E(PR-AUC)= {df_eval[colums[count]].mean():.2f}, \u03C3(PR-AUC)= {df_eval[colums[count]].std():.3f}"

        draw_brace(ax[m, n],
                   df_eval[colums[count]].mean(),
                   ax[m, n].get_ylim()[1] * 1.05,
                   stats_text)

        count += 1
        ax[m, n].spines['top'].set_visible(False)
        ax[m, n].spines['right'].set_visible(False)

    fig.tight_layout()
    # plt.show()

    plt.savefig(Path.cwd() / 'graphs/model_outputs/seed_lgb_10000_seed_distributions.pdf')
