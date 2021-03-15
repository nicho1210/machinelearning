#data preprocessing

#importing the dataset
dataset = read.csv('50_Startups.csv')
#categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))


#splitting the dataset into the Train set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Profit ~ R.D.Spend, 
               data = training_set)

y_pred = predict(regressor, newdata = test_set)


regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = training_set)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = training_set)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = training_set)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)
