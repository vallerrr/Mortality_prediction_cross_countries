import numpy as np
import pandas as pd
import itertools
import DataImport
import matplotlib.pyplot as plt
from Models import Model_fixed_test_size
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score


def domain_iterations_exhausted_combination(domain_to_iter, model_eval, iterate_num):
    iterations = list(itertools.combinations(domain_to_iter, iterate_num))
    for additional_domain in iterations:
        all_domains = ["demographic", "adult_SES"] + list(additional_domain)
        print('domain:  {}'.format(all_domains))

        # retrieve domain_lst
        domain_lst = list(set(domains['demographic'] + domains['adult_SES']+['age']))
        for single_domain in additional_domain: domain_lst = list(set(domain_lst + domains[single_domain]))

        # model fit
        model = Model_fixed_test_size(data=df, test_size=test_size, domain_list=domain_lst, model='xgb',
                                      train_subset_size=1, order=0)
        pred_prob, pred_label = model.test_set_predict_prob, model.test_set_predict
        y_test, sample_weight = model.y_test, model.test_sample_weight

        # pr part
        precision, recall, _ = precision_recall_curve(y_test, pred_prob, sample_weight=sample_weight)
        pr_f1, pr_auc = f1_score(y_test, pred_label, sample_weight=sample_weight), auc(recall, precision)
        pr_no_skill = len(y_test[y_test == 1]) / len(y_test)

        # roc
        roc_no_skill = 0.5
        auc_score = roc_auc_score(y_test, pred_prob, sample_weight=sample_weight)

        # domain name

        domain_name, switch = '', True
        for domain in all_domains:
            if (domain == 'demographic') or (domain == 'adult_SES'):
                continue
            else:
                if switch:
                    domain_name += domain + '+'
                    switch = False
                else:
                    domain_name += domain + '\n'
                    switch = True
        if domain_name.endswith('+'): domain_name = domain_name[0:len(domain_name) - 1]
        if domain_name =='': domain_name = 'basic'

        # columns=['model', 'domain_list', 'domain_num', 'domain_names','pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc'])
        model_eval.loc[len(model_eval)] = [model, all_domains, len(all_domains), domain_name, pr_no_skill, pr_f1, pr_auc,
                                           roc_no_skill, auc_score]
    model_eval.loc[model_eval['domain_num']==7,'domain_names'] = 'all'
    return model_eval




test_size = 0.3
df = DataImport.data_reader()
domains = DataImport.domain_dict()

# data structure to store info
model_eval = pd.DataFrame(
    columns=['model', 'domain_list', 'domain_num', 'domain_names',
             'pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc'])

# create domain list to iterate
domain_to_iter = list(domains.keys())
domain_to_iter.remove('demographic')
domain_to_iter.remove('adult_SES')
domain_to_iter.remove('all')

for iter_num in np.arange(0, 6, 1):
    model_eval = domain_iterations_exhausted_combination(domain_to_iter, model_eval, iter_num)

indicator = 'pr_auc'
model_eval.sort_values(indicator, inplace=True)



# plot part ----------------------------------
# pos : the y-axis position of each outcome
pos = [0]
pos_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 3}
colors = {2: "#C8E3D4", 3: "#87AAAA", 4: "#DADDFC", 5: '#577BC1', 6: "#506D84", 7: "#2E4C6D"}

# adjust the position and width of each bar
width = []
for i in np.arange(0, len(model_eval['domain_num']), 1):
    num = model_eval['domain_num'][i]
    pos.append(pos[-1] + pos_dict[str(num)])
    width.append(pos_dict[str(num)]*0.6)
pos = pos[1:len(pos)]

# plot
y = model_eval[indicator]
fig, ax = plt.subplots(figsize=(12, 22))
plt.xlim(min(y)-0.05, max(y)+0.02)
plt.barh(pos, y, align='center', height=width, color=model_eval['domain_num'].map(colors))
fig.subplots_adjust(left=0.2)
ax.set_xlabel(indicator)
handles = []
for i in pos_dict.keys():
    handles.append(mpatches.Patch(color=colors[int(i)], label=i))
plt.legend(handles=handles, loc=4, fontsize=18, title='Counts of Domains')
plt.yticks(pos, model_eval['domain_names'])

fig.tight_layout()
#plt.gca().invert_yaxis()
plt.show()
# end here----------------------------------



