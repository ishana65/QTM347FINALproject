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

The Dataset was originally in ARFF form and was converted to a CSV file using this code: 

```python
 #convert to csv
data, meta = arff.loadarff('/Users/natashagandhi/Desktop/QTM 347/speeddating.arff')
#convert to DataFrame
data = pd.DataFrame(data)
file_path = '/Users/natashagandhi/Desktop/speeddating.csv'
data.to_csv(file_path)
```

After cleaning, the subset 'numerical_subset_clean', containing only the numerical variables, was created. From there, the dataset 'rating_partner' was created and used for much of the analysis. 

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

In the next section, we will further explore which variables are important and influence our target variable, 'like. ' 

# Feature Selection: 

Using Random Forest and bagging, we looked at which variables were the most important using the feature importance function. We optimized our m values by plotting Train and Test MSE as M increased. First, we looked at all the variables against 'like.' 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/464fc721-2b6e-427b-991a-eeee13a992f3)

```python
m = best_m # the best m based on test error 
rf = RF(max_features=m, random_state=0).fit(X_train, y_train)
feature_imp = pd.DataFrame({'importance':rf.feature_importances_}, index=feature_names)
feature_imp.sort_values(by='importance', ascending=False)
```

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/35a3e2be-d50f-4dfb-9995-dcc0a9c91093)

According to this graph, many of the interest variables (clubbing, tv, tvsports, etc.) are at the bottom of the feature importance graph. Therefore, we decided not to include these in our variables of interest in later analysis. 

Furthermore, we want to discover whether similar features show up in our decision vs. our partners decision.

We want to look at the variable 'decision_o, ' which is the partner's decision on the night of the event. We want to see which features influence your partner's decision. We've included both the variables of partners' scoring preference of the different features (how highly they value attractiveness, intelligence, etc.) and the variables that show how your partner rated you on the night of the event. 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/32f7e004-7818-4169-afa5-848423e7f67c)

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/ad945eab-a74a-4662-b899-611157b8841e)

Our top 2 features are attractive_o, which is how attractive your partner rates you and shared_interests_o. This shows us that attractiveness and shared interests are the most important features in determining a partners decision during speed dating. 

Now, we want to examine 'decision', which includes features such as how you rated your partner on the night of the event and your preferences (such as how important attractiveness, sincerity, etc.). 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/fb37fe96-ffb5-4645-838c-7dcf7c6e4672)

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/96f5a91f-cff7-4281-aa69-cd6e41d64552)

It looks like the same traits are important in' decision'! Attractiveness and Shared Interests are the most important features in 'decision'. This finding is also consistent with our findings from the best subset selection. This is important since it can tell us that the features that are important to us are also important to our partner!

Due to this, we have decided to simplify our analysis going forward and only include the variables in the rating_partner dataset. We concluded that we would get similar results if using the variables that relate to partner ratings and importance. 

# Best Subset Selection: 

In running Best Subset Selection, this is how we split our testing and training data: 

```python
X = rating_partner[['importance_same_race',
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

y = rating_partner['like']

#split data 80% testing, 20% training                                        
test_size = int(rating_partner.shape[0] * 0.2)
rating_partner_train, rating_partner_test = train_test_split(rating_partner, test_size=test_size, random_state=0)


#define training data
y_train = rating_partner_train['like']
X_train = rating_partner_train[['importance_same_race',
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
#add a constant
X_train_cons = sm.add_constant(X_train)


#split data
#define testing data
y_test = rating_partner_test['like']
X_test = rating_partner_test[['importance_same_race',
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

#add a constant
X_test_cons = sm.add_constant(X_test)
```
These are the results of the Best Subset Selection: 

<img width="751" alt="Screenshot 2024-04-25 at 4 34 21 PM" src="https://github.com/ishana65/QTM347FINALproject/assets/122471733/25b88c19-8f82-441b-9f54-c2c014c12d90">

In the above best subset selection output, we can see that the top three predictors of like include attractive_partner, shared_interests_partner, and funny_partner. Training MSE = 1.153418584, Test MSE = 1.3683646. These three predictors are the same variables that had the highest correlation to like in the correlation matrix and the highest importance in the feature importance graphs. 

# Lasso Regularization: 

For Lasso, we started with standardizing out data and then using ElasticNetCV to find the optimal alpha value for Lasso. 

```python
Y = np.array(rating_partner['like'])
design = MS(rating_partner.columns.drop('like')).fit(rating_partner)
D = design.fit_transform(rating_partner)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

K=5
kfold = skm.KFold(K, random_state=0, shuffle=True)
lassoCV = skl.ElasticNetCV(n_alphas=100, 
                           l1_ratio=1, 
                           cv=kfold)
scaler = StandardScaler(with_mean=True, with_std=True)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
tuned_lasso.alpha_
```
After finding the optimal alpha value, we ran the Lasso regression using this code: 

```python
lasso = skl.ElasticNet(alpha= tuned_lasso.alpha_ 
                       , l1_ratio=1)
scaler = StandardScaler(with_mean=True, with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)]) 
pipe.fit(X_train, y_train)

coefficients = dict(zip(X_train.columns, lasso.coef_))
lasso_table = pd.DataFrame({'Feature': list(coefficients.keys()), 'Coefficient': list(coefficients.values())})
lasso_table_sorted = lasso_table.sort_values(by='Coefficient', ascending=False)

print(lasso_table_sorted.to_string(index=False))
```
The regression yielded these coefficients: 
                   Feature  Coefficient
        attractive_partner     0.622564
  shared_interests_partner     0.480673
             funny_partner     0.367187
          guess_prob_liked     0.240499
      intelligence_partner     0.190104
           sincere_partner     0.135778
      attractive_important     0.006206
                     d_age    -0.000000
                  samerace     0.000000
          ambition_partner     0.000000
         sincere_important     0.000000
           funny_important    -0.000000
        ambition_important    -0.000000
shared_interests_important     0.000000
    intelligence_important    -0.010303
      importance_same_race    -0.025238

According to Lasso, the three variables that most positively influence our target variable are attractive_partner, shared_interests_partner, and funny_partner. The Train MSE was 1.0312009921697962, and our Test MSE was 1.1564421955496138. These were the same results seen in the earlier models, further verifying the importance of these variables. 

We also plotted the coefficients against standardized lambda to see the path of the different variables. 

![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/8b018452-fc52-4ed5-9c17-f1677053ecc9)

# Decision Tree: 

The final step of our project was to create a decision tree based on the dataset. Ultimately, our goal was to transform this into an interactive algorithm that could be used by our peers. 

```python
np.random.seed(123)

reg = DTR()
reg.fit(X_train, y_train)

#test MSE
np.mean((y_test - reg.predict(X_test))**2)
```
Our original Test MSE was 2.2117224880382773 and Train MSE was 0.0. After pruning using this code:

```python
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train) 
num_arangees_per_splits = np.arange(2,11,2)
num_min_samples_leaf = range(1,6)


param_grid = {'ccp_alpha': ccp_path.ccp_alphas, 
              'min_samples_split': num_arangees_per_splits, 
              'min_samples_leaf': num_min_samples_leaf
             }


kfold = skm.KFold(5,
                  shuffle=True,
                  random_state=0) 


grid = skm.GridSearchCV(reg,
                        param_grid,
                        refit=True,
                        cv=kfold, 
                        scoring='neg_mean_squared_error')


G = grid.fit(X_train, y_train)

best_ = grid.best_estimator_
print(grid.best_params_, np.mean((y_test - best_.predict(X_test))**2))
```
Our new Test MSE was 1.710249225443976, which is lower than the original Test MSE, showing that pruning decreased the MSE of our Decision Tree. 

Pruned Tree: 

```python
#plot tree
ax = subplots(figsize=(80,80))[1] 

plot_tree(best_, 
          feature_names=X_train.columns, 
          ax=ax);
```
![image](https://github.com/ishana65/QTM347FINALproject/assets/122471733/09b7b73f-b223-4238-bd84-35e69cb4b67f)

According to the pruned tree, the features that are considered important are funny_partner, attractive_partner, and shared_interest_partner. These features are included in the top splits of the tree and are therefore considered more important. Some of the other variables that were selected include intelligence_partner, guess_prob_liked, ambition_partner, funny_important, importance_same_race, sincere_partner, sincere_important,  samerace, and intelligence_important.

# Model Comparison 

<img width="805" alt="Screenshot 2024-05-01 at 4 21 15 PM" src="https://github.com/ishana65/QTM347FINALproject/assets/122471733/c2943571-0f96-4e07-831b-ecd8b9c33fbd">

Comparing the 3 models (not considering Random Forest since different variables were used), we see that the Lasso performed the best based on Test MSE. This could be because Lasso reduces overfitting by shrinking the "non-important" coefficients to 0. With such a large dataset, the Lasso model performs well by reducing the noise and only considering the top coefficients. The Decision tree had the highest Test MSE. This suggests that despite the pruning, there was probably overfitting to the training data, and less important variables were used.

# Algorithm + App: 
Next, we wanted to convert this decision tree to an interactive algorithm. To do this, we created a web application using Streamlit, which can be accessed using the link below. Once on the application, users can answer a series of questions about their personal preferences and opinions of their partner, and when they click “calculate”, the algorithm will follow the decision tree and result in a like score, which is the “value” on the decision tree. The goal of this application is to provide information about how much someone may like their partner. Of course, this like score is not entirely accurate as several other factors must be considered, but it gives us a good understanding of how our opinions and preferences in a partner can influence how much we like them!

The Link to the App! 
https://qtm-347-project-2.onrender.com/

# References: 

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1.
		 	 	 		
Greitemeyer, Tobias. “What Do Men and Women Want in a Partner? Are Educated Partners Al- ways More Desirable?” Journal of Experimental Social Psychology, Academic Press, 29 Mar. 2006, www.sciencedirect.com/science/article/abs/pii/S0022103106000345. 

Hastie, T., Tibshirani, R., & Wainwright, M. (2015). Statistical Learning with Sparsity: The Lasso and Generalizations. CRC Press.

