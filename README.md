# GLPGDSBA_Dec21_Hackathon
I recently participated in the Shinkansen Travel Experience - Hackathon organized by Great Learning for PG-DSBA course attendees. This is my experience with the Hackathon. It was an exciting two days event and enormous learning opportunity for like minded Machine Learning enthusiasts .
## Below is the Problem Description 
This is the problem of a Shinkansen (Bullet-Trains) of Japan. They aim to determine the relative importance of each parameter with regards to their contribution to the passenger travel experience. Provided is a random sample of individuals who travelled using their train. The on-time performance of the trains along with the passenger’s information is published in the CSV file named ‘Traveldata_train’.  These passengers were later asked to provide their feedback on various parameters related to the travel along with their overall experience. These collected details are made available in the survey report CSV labelled ‘Surveydata_train’.

In the survey, a passenger was explicitly asked whether they were delighted with their overall travel experience and that is captured in the data of the survey report under the variable labelled ‘Overall_Experience’. 

The objective of this exercise is to understand which parameters play an important role in swaying passenger feedback towards a positive scale. You are provided test data containing Travel data and Survey data of passengers. Both the test data and the train data are collected at the same time and belongs to the same company.
### Goal:
The goal of the problem is to predict whether a passenger was delighted considering his/her overall travel experience of traveling in Shinkansen (Bullet Train). For each passenger id in the test set, you must predict the “Overall_Experience” level.

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

Both Departure Delay and Arrival Delay columns are numeric columns and have high no. of NULLs and the distribution is highliy skewed - #### hence we cannot use impute using Mean. I impute both the columns using their Median values. 
Age is a numeric variable but the distribution is Normal , hence I impute Age with the Mean value. 
All categorical variables were imputed using mode. Imputation is a very subjective issue and requires domain expertise. Here since its a hackathon and the aim is to achieve high Accuracy scores, I tried out the copy book method of imputation.

#### Next comes an important step to convert all the Categorical variables to Integers as is required by Python's sklearn library for all Machine Learning algorithms. 
Here since, the values are Survey attributes, we need to be very sure of assigning exact same orders and hence I took the hardcoding route to assign numbers to the Survey variables instead of using Pandas Categorical converter.

Once all missing values are imputed and Categoricals converted to numeric, the data is ready for model building.

### Model Building

First step of model building is to split the data into train and test. I use a 70-30 random split using sklearn train test split .
The below are the Models built:
Naive Bayes
KNN
Random Forest with and without Hyper Parameter tuning
Bagging with and without without Hyper Parameter tuning
Adaboost with and without without Hyper Parameter tuning
Gradient Boost with and without Hyper Parameter tuning
LDA with and without without Hyper Parameter tuning
Logistic Regression with and without without Hyper Parameter tuning

One important point here is about scaling - Since this is a Survey data already in a scale of 1-10 , and only 2 numeric variables exists, I decided not to scale the data. Moreover scaling is needed mostly for LDA/Log Regression but tree based algos work fine without scaling. Hence I took the chance.

Finally ! the Algorithm that worked magic for me - XGBoost 



****
