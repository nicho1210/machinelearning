#data preprocessing

#importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]
#splitting the dataset into the Train set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#fitting simple linear regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

#predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set),
            colour = 'green')) +
  ggtitle ('Salary V.S. Experience(Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary') 

   
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'blue') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set),
                colour = 'green')) +
  ggtitle ('Salary V.S. Experience(Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary') 



