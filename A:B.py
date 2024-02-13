#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as ss
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


groups = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-anton-ivanov/Проект_2_groups.csv', sep=';')
groups_add = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-anton-ivanov/Проект_2_group_add.csv')
active_studs = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-anton-ivanov/Проект_2_active_studs.csv')
checks = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-anton-ivanov/Проект_2_checks.csv', sep=';')


# Провожу предварительный анализ данных

# In[ ]:





# In[3]:


groups = pd.concat([groups, groups_add])


# In[4]:


groups


# In[5]:


active_studs_id = active_studs.student_id


# In[6]:


active_stud_grp = groups.query('id in @active_studs_id')


# In[7]:


active_stud_grp = active_stud_grp.rename(columns={'id':'student_id'})


# In[8]:


active_stud_grp.head()


# In[9]:


checks.head()


# In[10]:


checks_id = checks.student_id


# In[11]:


checks_id.shape[0]


# Объединяю датафреймы для получения активных студентов в дни теста и их покупки или отсутствие покупок

# In[12]:


active_stud_full = active_stud_grp.merge(checks, how='left', on='student_id')
active_stud_full


# In[13]:


active_stud_full.dtypes


# In[14]:


active_stud_full.grp = active_stud_full.grp.apply(str)


# In[50]:


active_stud_full['target_action'] = active_stud_full.query('rev != "NaN"').student_id.isin(active_stud_full.student_id).astype(int)
active_stud_full.target_action = active_stud_full.target_action.fillna(0)


# In[51]:


grp_a = active_stud_full[active_stud_full.grp == 'A']
grp_b = active_stud_full[active_stud_full.grp == 'B']


# In[52]:


grp_a.student_id.nunique()


# In[53]:


grp_b.student_id.nunique()


# In[54]:


groups[groups.grp == 'B'].id.nunique()


# In[55]:


groups[groups.grp == 'A'].id.nunique()


# Проверяю нет ли пересечений между группами студентов

# In[56]:


grp_1 = set(grp_a.student_id)
grp_2 = set(grp_b.student_id)


# In[57]:


grp_1.intersection(grp_2) # пересечений нет


# In[58]:


active_stud_full.rev.describe()


# In[59]:


grp_a.rev.describe()


# In[60]:


grp_b.rev.describe()


# In[ ]:





# Считаю целевые метрики. Конверсию в покупку и средний чек в группах. Выбираю эти метрики, т.к это основные метрики продукта, которые можно получить из представленных данных.

# In[61]:


#conversion rate - отношение общего количества пользователей, которые совершили покупку, к общему числу пользователей
cr_a = grp_a.query('rev != "Nan"').student_id.nunique() / grp_a.student_id.nunique()
cr_b = grp_b.query('rev != "Nan"').student_id.nunique() / grp_b.student_id.nunique()


# In[62]:


cr_b


# In[63]:


cr_a


# In[64]:


avg_rev_a = grp_a.rev.sum() / grp_a.query('rev != "Nan"').student_id.nunique()
avg_rev_b = grp_b.rev.sum() / grp_b.query('rev != "Nan"').student_id.nunique()


# In[65]:


avg_rev_a


# In[66]:


avg_rev_b


# In[113]:


sns.boxplot(data=active_stud_full, x='grp', y='rev')


# Предварительно можем увидеть значительное увеличение среднего и медианного чеков в тестовой группе.

# In[ ]:





# Проверяю было ли больше одной покупки на каждого уникального студента

# In[67]:


grp_a.groupby('student_id', as_index=False).agg({'rev': 'count'}).query('rev > 1') #нет


# In[68]:


grp_b.groupby('student_id', as_index=False).agg({'rev': 'count'}).query('rev > 1') #нет


# In[1]:


active_stud_full.head()


# In[ ]:





# **Приступаю к статистическим тестам.**  
# Гипотеза 1. Уменьшилась конверсия в покупку.

# In[ ]:





# In[97]:


ss.normaltest(grp_b.target_action) #проверяю распределения на нормальность


# In[98]:


ss.normaltest(grp_a.target_action)


# In[135]:


pg.ttest(x=grp_b.target_action, y=grp_a.target_action, correction=True) #t-test


# In[101]:


pg.mwu(x=grp_b.target_action, y=grp_a.target_action) # Манна-Уитни


# In[99]:


cr_a


# In[100]:


cr_b


# Т.к p-value больше 0.05, то делаю вывод что статически значимых изменений в конверсии нет, несмотря на разницу в группах показанную выше

# In[ ]:





# Гипотеза 2. Средний чек увеличился

# In[ ]:





# In[122]:


sns.histplot(data=active_stud_full, hue='grp', x='rev') #распределение не нормальное


# In[136]:


sns.boxplot(data=active_stud_full, x='grp', y='rev')


# In[131]:


pg.ttest(x=grp_b.rev, y=grp_a.rev, correction=True)


# In[127]:


pg.mwu(x=grp_b.rev, y=grp_a.rev)


# In[132]:


avg_rev_a


# In[134]:


avg_rev_b


# p-value в t-test выше, но очень близко к пороговому значению 0.05, а в Манна-Уитни значение значительно ниже порогового. Также занчительная разница в средних и медианных значениях чеков представленных выше позволяет мне прийти к выводу, что изменения с средних чеках статистически значимы.

# In[ ]:





# Итого: На основании проведенного анализа считаю, что новую механику оплаты можно запускать на всех пользователей, т.к средний чек статистически значимо увеличился, а cr не изменился, следовательно изменения могут привести к увеличению выручки.
