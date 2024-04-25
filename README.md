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

`` #convert to csv
data, meta = arff.loadarff('/Users/natashagandhi/Desktop/QTM 347/speeddating.arff')
#convert to DataFrame
data = pd.DataFrame(data)
file_path = '/Users/natashagandhi/Desktop/speeddating.csv'
data.to_csv(file_path) ``

After cleaning, the subset 'numerical_subset_clean' was created, containing only the numerical variables. From there, the dataset, 'rating_partner', was created and used for much of the analysis. 

``rating_partner = numerical_subset_clean[['like', 
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
``

Introductory Data Analysis: 


Feature Selection: 


Best Subset Selection: 


Lasso: 


Decision Tree: 


Algorithm + App: 

