# Exercise 7
# Topic: Modeling Consumer Behavior
# Alexander Haas (haas.alexanderjulian@gmail.com)

library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('rpart')
library('caret')

########################################################################################################
# TUTORIAL 1: https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
# Read data 
path = "~/Desktop/Uni/MACB/Exercises/Übung 7/"
train = read.csv(paste(path, "Data/train.csv", sep = ""), stringsAsFactors = F)
test = read.csv(paste(path, "Data/test.csv", sep = ""), stringsAsFactors = F)

# Merge train and test set
full  <- bind_rows(train, test) # bind training & test data

# check data
str(full)

# grab title from passenger name (http://www.endmemo.com/program/R/gsub.php: . = any character; * multiple times; (a|b) either a or b) 
# --> any character multiple times until ", " replace with "" (=delete); 
# --> any character after . 
full$Title = gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title)
rareTitle = c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full$Title[full$Title=="Mlle"] = "Miss"
full$Title[full$Title=="Ms"] = "Miss"
full$Title[full$Title=="Mme"] = "Mrs"
full$Title[full$Title %in% rareTitle]  = "Rare Title"

# grab surname from passenger name 
full$Surname = sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1]) 
cat(paste('We have <b>', nlevels(factor(full$Surname)), '</b> unique surnames (',dim(full),'people in total). I would be interested to infer ethnicity based on surname --- another time.'))

# family size
full$Fsize = full$SibSp + full$Parch + 1
full$Family <- paste(full$Surname, full$Fsize, sep='_')

ggplot(full[1:891,], aes(x=Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

full$FsizeD[full$Fsize == 1] = "singleton"
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] = "large"
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

# deck
strsplit(full$Cabin[2], NULL)[[1]]
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

embark_fare = full %>% filter(full$PassengerId!=62 & full$PassengerId!=830)

ggplot(embark_fare, aes(x=Embarked, y=Fare, fill=factor(Pclass))) +
  geom_boxplot() + 
  geom_hline(aes(yintercept=80), colour='black', linetype='dashed', lwd=0.5) + 
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

full$Embarked[c(62, 830)] <- 'C'

ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

sum(is.na(full$Age))

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)
# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

# First we'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()
  
# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# Show counts
table(full$Mother, full$Survived)

# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

md.pattern(full)

# Split the data back into a train set and a test set
train1 = train 
test1 = test
train <- full[1:891,]
test <- full[892:1309,]

# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)

# Show model error
par(mfrow=c(1,1))
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = '~/Desktop/Uni/MACB/Exercises/Übung 7/Solution/rf_mod_Solution.csv', row.names = F)

########################################################################################################
# TUTORIAL 2: https://www.kaggle.com/jasonm/large-families-not-good-for-survival 

# Use features from Tutorial1 and subsample training data in new training and test set 
# Subsample training data 
set.seed(820)
inTrainingSet = createDataPartition(train$Survived, p = 0.7, list=FALSE)
subtrain = train[inTrainingSet,]
subtest = train[-inTrainingSet,]

# Accuracy 
modelaccuracy <- function(test, rpred) {
  result_1 <- test$Survived == rpred
  sum(result_1) / length(rpred)
}

checkaccuracy <- function(accuracy) {
  if (accuracy > bestaccuracy) {
    bestaccuracy <- accuracy
    assign("bestaccuracy", accuracy, envir = .GlobalEnv)
    label <- 'better'
  } else if (accuracy < bestaccuracy) {
    label <- 'worse'
  } else {
    label <- 'no change'
  }
  label
}

# starting with Age and Sex as indicators
fol <- formula(Survived ~ Age + Sex)                        # 0.845
rmodel <- rpart(fol, method="class", data=subtrain)
rpred <- predict(rmodel, newdata=subtest, type="class")
accuracy <- modelaccuracy(subtest, rpred)
bestaccuracy <- accuracy # init base accuracy
print(c("accuracy1", accuracy))                             # baseline
