# GLPGDSBA_Dec21_Hackathon
I recently participated in the Shinkansen Travel Experience - Hackathon organized by Great Learning for PG-DSBA course attendees. It was an exciting two days event and enormous learning opportunity for like minded Machine Learning enthusiasts . Although it was an individual participation event, I named myself "Natural Intelligence" .Below is my experience with the Hackathon with some Nerdy facts on my approach.
## Below is the Problem Description 
This is the problem of a Shinkansen (Bullet-Trains) of Japan. They aim to determine the relative importance of each parameter with regards to their contribution to the passenger travel experience. Provided is a random sample of individuals who travelled using their train. The on-time performance of the trains along with the passenger’s information is published in the CSV file named ‘Traveldata_train’.  These passengers were later asked to provide their feedback on various parameters related to the travel along with their overall experience. These collected details are made available in the survey report CSV labelled ‘Surveydata_train’.

In the survey, a passenger was explicitly asked whether they were delighted with their overall travel experience and that is captured in the data of the survey report under the variable labelled ‘Overall_Experience’. 

The objective of this exercise is to understand which parameters play an important role in swaying passenger feedback towards a positive scale. You are provided test data containing Travel data and Survey data of passengers. Both the test data and the train data are collected at the same time and belongs to the same company.
### Goal:
The goal of the problem is to predict whether a passenger was delighted considering his/her overall travel experience of traveling in Shinkansen (Bullet Train). For each passenger id in the test set, you must predict the “Overall_Experience” level.

### Evaluation Criteria for the Hackathon : Accuracy metric.

### Dataset

The problem consists of 2 separate datasets: Travel data & Survey data. The Travel data has information related to passengers and the performance of the Train in which they traveled. The survey data is the aggregated data of surveys collected post-service experience. You are expected to treat both the datasets as raw data and perform any necessary cleaning/validation steps as required.

## My Approach

Converting a business problem statement into Machine Learning Problem statement is the most critical factor in success of Data Science initiatives. To be able to do that we need to be thorough with the data. I solved the problem with Python. So first step was to import and understand the data.  

Screenshot of the Travel Dataset: 
![image](https://user-images.githubusercontent.com/18433095/148481521-edb9e8b3-478b-4c52-bc16-3574a6f45f60.png)

Screenshot of the Survey Dataset:
![image](https://user-images.githubusercontent.com/18433095/148558426-b2e0580c-d0be-449b-b6d9-abc9d55ceca2.png)

A quick info of the dataset told that there were NULLs in both the datasets:
Seat_comfort                 61
Arrival_time_convenient    8930
Catering                   8741
Platform_location            30
Onboardwifi_service          30
Onboard_entertainment        18
Online_support               91
Onlinebooking_Ease           73
Onboard_service            7601
Leg_room                     90
Baggage_handling            142
Checkin_service              77
Cleanliness                   6
Online_boarding               6
Gender                       77
CustomerType               8951
Age                          33
TypeTravel                 9226
DepartureDelay_in_Mins       57
ArrivalDelay_in_Mins        357      

The Next thing I did was merged the datasets using ID column to have a unified look at the travel data including Survey results and Travel attributes.
#### Then I did EDA on the combined dataset. Below were the key findings of EDA:-
 Almost 70% of people who have responded OnboardWifi Service to be excellent have also given an overall rating of 1 which means they were delighted with the overall experience :
 
 ![image](https://user-images.githubusercontent.com/18433095/148559542-b23197d9-8266-4245-b57b-48ee363db009.png)
 
Cleanliness is another very distinct factor of overall satisfaction as can be seen from the below chart. 100% of people who have given  extremely poor for cleanliness were also dissatisfied with the overall journey.

![image](https://user-images.githubusercontent.com/18433095/148559655-cb13d45e-f5a0-404b-8277-95dd126fb0a8.png)

Similarly, online booking and online service have very clear impact on the Overall satisfaction:

![image](https://user-images.githubusercontent.com/18433095/148560260-76a2ffff-c110-4814-98f0-736041d642d9.png)

Another very important feature that impacts on the Overall experience based on Historical data is type of Traveller - More than 70% of business class travellers are satisfied with the journey overall whereas only around 40% of the Economy class are satisfied.
![image](https://user-images.githubusercontent.com/18433095/148561125-e2413ab9-4407-4c29-aa68-455562cfe8a6.png)

So we can already see many features which make an impact on the overall satisfaction of the users and can be used to predict whether future users will be satisfied and can be used to improve the overall satisfaction.

### Next I move to the Data Processing part

Both Departure Delay and Arrival Delay columns are numeric columns and have high no. of NULLs and the distribution is highliy skewed - #### hence we cannot use impute using Mean. I impute both the columns using their Median values. The Median Delay is 0 which means No delay , so the business assumption here is where there is No delay value in the data, the train has been on time. But in real life this needs to be validated by domain experts. 
Age is a numeric variable but the distribution is Normal , hence I impute Age with the Mean value. 
All categorical variables were imputed using mode. Imputation is a very subjective issue and requires domain expertise. Here since its a hackathon and the aim is to achieve high Accuracy scores, I tried out the copy book method of imputation.
##### One important point here is about scaling - Since this is a Survey data, already in a scale of 1-10 , and only 2 numeric variables exists, I decided not to scale the data. Moreover scaling is needed mostly for LDA, Log Regression and KNN but tree based algos work fine without scaling. Hence I took the chance of first trying without scaling and it worked well for me. But we can always try scaling the Numeric variables separately, especially when there more numeric variables.

#### Next comes an important step to convert all the Categorical variables to Integers as is required by Python's sklearn library for all Machine Learning algorithms. 
Here since, the values are Survey attributes, we need to be very sure of assigning exact same orders and hence I took the hardcoding route to assign numbers to the Survey variables instead of using Pandas Categorical converter.

Once all missing values are imputed and Categoricals converted to numeric, the data is ready for model building.

### Model Building

First step of model building is to split the data into train and test. I use a 70-30 random split using sklearn train test split .
Then, I built the below are the Models for Classification. Now for each of these models, I tried two types of dataset - 
      1. One with only a subset of columns that were more impactful to the Target variable as identified in EDA. 
      2. Other with all columns. 
Although I was getting good accuracy with the former with train data, I was getting lower than the other compettitors, when I submitted my results on Test data in the Hackathon portal. Hence I finally decided to go with all columns irrespective of their importance. 
###### In real life probably, the model trained on a subset of important columns will be better as we dont want our model to be overfit.
      
1. Naive Bayes
2. KNN
3. Random Forest with and without Hyper Parameter tuning
4. Bagging with and without without Hyper Parameter tuning
5. Adaboost with and without without Hyper Parameter tuning
6. Gradient Boost with and without Hyper Parameter tuning
7. LDA with and without without Hyper Parameter tuning
8. Logistic Regression with and without without Hyper Parameter tuning


##### Finally ! the Algorithm that worked magic for me - XGBoost. But- "there are No Free lunches" . All these tree based Algorithms come with a huge tax on Processing power of your system, especially when you are tuning them iteratively. Since I was using personal laptop for this excercise, it wasnt feasible to run too many iterations of RF, Bagging Boosting. So I did two iterations of XGBoost with manual tuning rather than Grid Search
The below are the parameters of XGBoost that I played around with to get the best model :
1. colsample_bytree : This parameter determines the fraction of features that is used to train each tree. In the default model it was 1, I changed it to 0.5 to make each tree weak learner so that cummulative learning can be strong.
2. learning_rate : This parameter tells you how quickly the model fits the residual errors by using additional base learners . Default was 0.300000012 I slowed it down to 0.1. Although its a risk when your processor is slow as the model would take smaller and more no. of steps to train in this case. But since I took this calculated risk.
3. n_estimators: This is another parameter where I took a risk of slowing down the trainging process because more trees definitely means better cummulative learning but also means more processing resource requirements. Default was 100, I changed to 200 trees.
4. subsample: Fraction of the training set that can be used to train each tree. Default was 1 , I chose 0.7 as I wanted each individual tree to be built only on a smaller subset and not the whole dataset.

##### And Here I was ! with the model that gave me the best accuracy amongst all other Models. So I went ahead and submitted it and Bingo ! This was best my accuracy amongst all my 8 submissions! 0.95 . Ofcourse ! I would have loved to do more feature engineering or build more models to feature in that premium Top 10 spot. But I thoroughly enjoyed the experience and learnt a lot. So to all you Data Science rookies out there -" Dont wait for the Blue sky , jump in to every opportunity that you see to hone your skills " . 

![image](https://user-images.githubusercontent.com/18433095/148679099-dffeb04b-d1f4-4429-a9aa-6a3de05977f9.png)

****
