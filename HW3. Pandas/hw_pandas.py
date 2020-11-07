#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg">
# ## Открытый курс по машинному обучению
# <center>
# Автор материала: Юрий Кашницкий, программист-исследователь Mail.Ru Group <br> 
# 
# Материал распространяется на условиях лицензии [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Можно использовать в любых целях (редактировать, поправлять и брать за основу), кроме коммерческих, но с обязательным упоминанием автора материала.

# # <center>Домашнее задание № 1 (демо).<br> Анализ данных по доходу населения UCI Adult</center>

# **В задании предлагается с помощью Pandas ответить на несколько вопросов по данным репозитория UCI [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) (качать данные не надо – они уже есть в репозитории).**

# Уникальные значения признаков (больше информации по ссылке выше):
# - age: continuous.
# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - fnlwgt: continuous.
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - education-num: continuous.
# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - sex: Female, Male.
# - capital-gain: continuous.
# - capital-loss: continuous.
# - hours-per-week: continuous.
# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.   
# - salary: >50K,<=50K

# In[203]:


import pandas as pd

pd.set_option('display.max_rows', 100)
pd.__version__


# In[195]:


data = pd.read_csv('adult.data.csv')
data.head()


# In[6]:


data.info()


# **1. Сколько мужчин и женщин (признак *sex*) представлено в этом наборе данных?**

# In[8]:


pd.unique(data['sex']).tolist()


# **2. Каков средний возраст (признак *age*) женщин?**

# In[11]:


data[data['sex'] == 'Female']['age'].mean()


# **3. Какова доля граждан Германии (признак *native-country*)?**

# In[30]:


data['native-country'].value_counts(normalize = True)['Germany']


# **4-5. Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает более 50K в год (признак *salary*) и тех, кто получает менее 50K в год? **

# In[133]:


#v1
std1 = data[data['salary'] == '>50K']['age'].std()
std2 = data[data['salary'] == '<=50K']['age'].std()
print('std for >50K:', std1)
print('std for <=50K:', std2)

#v2
data.groupby(['salary'])['age'].agg(pd.DataFrame.std)


# **6. Правда ли, что люди, которые получают больше 50k, имеют как минимум высшее образование? (признак *education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters* или *Doctorate*)**

# In[75]:


higher_educ = ('Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate')
uniq_educ_more50 = data[data['salary'] == '>50K']['education'].unique()
res = False

for i in uniq_educ_more50:
    if(i not in higher_educ):
        res = True
        break
        
print("It's", res)


# **7. Выведите статистику возраста для каждой расы (признак *race*) и каждого пола. Используйте *groupby* и *describe*. Найдите таким образом максимальный возраст мужчин расы *Amer-Indian-Eskimo*.**

# In[61]:


data.groupby(['race'])['age'].describe()


# In[74]:


data[data['race'] == 'Amer-Indian-Eskimo']['age'].max()


# **8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или холостых мужчин (признак *marital-status*)? Женатыми считаем тех, у кого *marital-status* начинается с *Married* (Married-civ-spouse, Married-spouse-absent или Married-AF-spouse), остальных считаем холостыми.**

# In[132]:


#v1
married_count = len(data[(data['salary'] == '>50K') & (data['marital-status'].str.contains('Married'))])
ummarried_count = len(data[(data['salary'] == '>50K')]) - married_count
print('married:', married_count)
print('ummarried:', ummarried_count)


# In[174]:


#v2
mar_status_val_counts = data[data['salary'] == '>50K']['marital-status'].value_counts()
married_count = mar_status_val_counts[mar_status_val_counts.index.str.contains('Married')].sum()
ummarried_count = mar_status_val_counts.sum() - married_count
print('married:', married_count)
print('ummarried:', ummarried_count)


# **9. Какое максимальное число часов человек работает в неделю (признак *hours-per-week*)? Сколько людей работают такое количество часов и каков среди них процент зарабатывающих много?**

# In[183]:


print('max hours per week:', data['hours-per-week'].max())
print('number of people:', len(data[data['hours-per-week'] == data['hours-per-week'].max()]))
pd.crosstab(data[data['hours-per-week'] == data['hours-per-week'].max()]['hours-per-week'],  data['salary'], normalize = True)


# **10. Посчитайте среднее время работы (*hours-per-week*) зарабатывающих мало и много (*salary*) для каждой страны (*native-country*).**

# In[215]:


#v1
data.groupby(['native-country', 'salary'])['hours-per-week'].mean()


# In[213]:


#v2
data.pivot_table(['hours-per-week'], ['native-country', 'salary'], aggfunc='mean')

