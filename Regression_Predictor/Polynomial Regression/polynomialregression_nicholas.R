#data preprocessing

#importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[ , 2:3]
#dataset = dataset[, 2:3]
#splitting the dataset into the Train set and Test set
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

#Fitting linear regression to the dataset
lin_reg = lm(formula = Salary~. , 
             data = dataset)

#Fitting polynomial regression to the dataset
dataset$level2 = dataset$Level^2
dataset$level3 = dataset$Level^3
dataset$level4 = dataset$Level^4
poly_reg = lm(formula = Salary~. , 
             data = dataset)

#visualising the linear regression results(blue)
#visualising the linear regression results(green)
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue')+
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'black')+
  xlab('Level')+
  ylab('Salary')

#predicting a new result with linear regression
y_pred = predict(lin_reg,data.frame(Level = 6.5)) 
                 
#predicting a new result with polynomial regression
y_polypred = predict(poly_reg,data.frame(Level = 6.5,
                                     level2 = 6.5^2,
                                     level3 = 6.5^3,
                                     level4 = 6.5^4))
