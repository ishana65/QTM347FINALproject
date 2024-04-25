## QTM 347 FINAL Project - Love Connection
In this project, we’re exploring the dynamics of human compatibility and attraction in relationships. We are using speed-dating data to better understand the qualities and personalities that foster connections and mutual attraction. As young adults navigating modern relationships, we hope to gain insights that enrich our experiences and help us build more meaningful and fulfilling relationships.

We intend to conduct this analysis by using the following models:
 - Random Forest w/ Bagging 
 - Best Subset Selection 
 - Lasso Regularization 
 - Decision Tree Regressor

# Description of the Raw Dataset 
We used the OpenML Speed Dating Dataset ((https://www.openml.org/search?type=data&status=active&sort=nr_of_downloads&id=41), which includes data from experimental 4-minute speed dating events from 2002 to 2004. The dataset includes 123 features, including participant demographic information, partner characteristics, and compatibility. 

<img width="941" alt="Screenshot 2024-04-25 at 1 18 26 PM" src="https://github.com/ishana65/QTM347FINALproject/assets/122471733/a9651703-b96f-4e4b-b185-04cc1f4a85e3">

<img width="947" alt="Screenshot 2024-04-25 at 1 19 18 PM" src="https://github.com/ishana65/QTM347FINALproject/assets/122471733/97c6895a-4430-4287-9c04-26b58a7dbf10">

<img width="996" alt="Screenshot 2024-04-25 at 1 20 10 PM" src="https://github.com/ishana65/QTM347FINALproject/assets/122471733/344d3c09-b1fa-47bf-aaaf-5729684d0581">

Variables that end in _partner and _important refer to your rating of your partner on the night of the speed dating event and how important you rate each of the features, while variables that include _o refer to your partner’s rating of you as well as the importance your partner places on these characters. 

# Data Cleaning 

The Dataset was originally in ARFF form and was converted to a csv file using this code: 

```python
 #convert to csv
data, meta = arff.loadarff('/Users/natashagandhi/Desktop/QTM 347/speeddating.arff')
#convert to DataFrame
data = pd.DataFrame(data)
file_path = '/Users/natashagandhi/Desktop/speeddating.csv'
data.to_csv(file_path)
```

After cleaning, the subset 'numerical_subset_clean' was created, containing only the numerical variables. From there, the dataset, 'rating_partner', was created and used for much of the analysis. 

``` python
rating_partner = numerical_subset_clean[['like', 
                                         'importance_same_race',
                                         'd_age', 
                                         'samerace',
                                         'attractive_partner', 
                                         'sincere_partner', 
                                         'intelligence_partner', 
                                         'funny_partner', 
                                         'ambition_partner', 
                                         'shared_interests_partner', 
                                         'attractive_important', 
                                         'sincere_important', 
                                         'intelligence_important', 
                                         'funny_important', 
                                         'ambition_important', 
                                         'shared_interests_important', 
                                         'guess_prob_liked']]
```

# Introductory Data Analysis: 

In this section, we will talk about some of the Introductory Data Analysis that we conducted on our variables of interest. First, we wanted to explore how people view or rate themselves vs. how their partners rate them. This graph was yielded. 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/890aff65-771d-4600-9aab-e4d014fc8cbb)

In the above plot, it's interesting to see that people tend to rate themselves higher than their partner rates them across all five characteristic variables. This highlights bias in ratings and is something we will keep in mind during our data exploration.

Secondly, we wanted to examine if the variables decision (0/1) and like (1-10) were related. 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/99f5cf7a-8ce3-4500-b97b-4eb8f21eb008)

In the above violin plot, we can see that some decisions are made regardless of how much they 'like' their partner. For all ratings of like, even a 10, some participants still decided they did not want to further their connection. However, it appears that a partner needed to be liked at least a 4 to be given a yes. So, people's ultimate decisions seem to be somewhat unpredictable, and we will, therefore, use 'like' as our decision variable as we feel this is a better indicator of the strength of the connection. 

Next, we wanted to explore the differences between Preference in Males and Females. 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/d13530e7-93f0-42d5-b1a8-911bcccf5cab)

The plot above shows variations in characteristic importance based on gender. On average, men tend to place higher importance on partner attractiveness, and women tend to place higher importance on partner sincerity, humor, and ambition.  

This male finding is consistent with previous research outlined in the paper, 'What do men and women want in a partner? Are educated partners always more desirable?' by Tobias Greitemeyer. Here, it's said that men's preference for female attraction can be explained by evolutionary theories, including natural/sexual selection. Specifically, he says 'Thus, the adaptive problem for men is to identify women with high reproductive potential. Since a female’s beauty is associated with her fertility (Buss & Barnes, 1986), men value physical attractiveness more than women.' It's interesting to see this consistency within our data.

Correlation Plots: 

```python
pd.plotting.scatter_matrix(rating_partner, figsize=(40,40))
plt.show()
```
![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/13d1e4ef-68e6-4934-831a-dec9b187c5f6)

```python
correlation_matrix = rating_partner.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/923d7248-194b-4002-bcff-ee5f2da8d980)

Looking at the correlation matrix, we can see which variables highly correlate with our desired target variable, 'like.' Funny_partner, Attractive_partner, and shared_interests_partner are the most highly correlated with like.  

# Feature Selection: 


# Best Subset Selection: 


# Lasso: 


# Decision Tree: 


# Algorithm + App: 

