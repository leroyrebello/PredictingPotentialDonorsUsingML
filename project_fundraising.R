##Load the fundraising file

fundraising.df <- read.csv(file.choose(), header = T)
View(fundraising.df)
head(fundraising.df)
summary(fundraising.df)
dim(fundraising.df)
str(fundraising.df)

fundraising.df <- fundraising.df[,-c(24)]


# This data set has 3120 rows and 24 columns summarizing information about Fundraising.
# After removing the variables, dataset has 3120 rows and 18 columns summarizing information about Fundraising.

#. k-NN Classification 

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 1: look into columns to finalize which columns to use in Model

# Columns to use
data.frame(mean=sapply(fundraising.df, mean), 
           sd=sapply(fundraising.df, sd), 
           min=sapply(fundraising.df, min), 
           max=sapply(fundraising.df, max), 
           median=sapply(fundraising.df, median), 
           length=sapply(fundraising.df, length),
           miss.val=sapply(fundraising.df, function(x) 
             sum(length(which(is.na(x))))))
str(fundraising.df)

#The variable of interest here will be TARGET_B

# correlation 
cor(fundraising.df$TARGET_B,fundraising.df,use="complete.obs")

# x and y values -c(1,2,14,16,19,24)
#x <- c(3:13,15,17)
x <- c(1:22)
y <- 23


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 2: Partition Dataset into 60% Train,30% Validation  and 10% test Split

set.seed(1)

## partitioning into training (60%) validation (30%) and test (10%)
# randomly sample 60% of the row IDs for training; the remaining 30% serve as validation and 10% for testing
train.rows <- sample(rownames(fundraising.df), dim(fundraising.df)[1]*0.6)
# collect all the columns with training row ID into training set:
train.df <- fundraising.df[train.rows, ]
# assign row IDs that are not already in the training set, into validation 
valid.rows <- sample(setdiff(rownames(fundraising.df), train.rows), 
                     dim(fundraising.df)[1]*0.3)  
valid.df <- fundraising.df[valid.rows, ]

test.rows <- setdiff(rownames(fundraising.df), union(train.rows, valid.rows))
test.df <- fundraising.df[test.rows, ]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 3: Try with normalized dataset
# initialize normalized training, validation data, assign (temporarily) data frames to originals

train.norm.df <- train.df
valid.norm.df <- valid.df
test.norm.df <- test.df
fundraising.norm.df <- fundraising.df

head(train.norm.df)
summary(train.norm.df)

install.packages("ggplot2")
install.packages("lattice")
install.packages("caret")
install.packages("PreProcess")
library(caret)
norm.values <- preProcess(train.df[, x], method=c("center","scale"))
head(norm.values)
train.norm.df[, x] <- predict(norm.values, train.df[, x])
head(train.norm.df)

##Similarly scale valid data and fundraiser data
valid.norm.df[, x] <- predict(norm.values, valid.df[, x])
fundraising.norm.df[, x] <- predict(norm.values, fundraising.df[, x])

test.norm.df[, x] <- predict(test.values, valid.df[, x])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 4: Final k-NN model with k = 3
# use knn() to compute knn. 
# knn() is available in library FNN (provides a list of the nearest neighbors) 

library(FNN)
# Normalized Dataframe
fundraisingKnnnorm <- knn(train = train.norm.df[, x], test = valid.norm.df[, x], 
                       cl = train.norm.df$TARGET_B, k = 3) 
dim(train.norm.df)
fundraisingKnn <- knn(train = train.df[, x], test = valid.df[, x], 
                   cl = train.df[, y], k = 3) 

fundraisingKnnnorm1 <- knn(train = train.norm.df[, x], test = test.norm.df[, x], 
                          cl = train.norm.df$TARGET_B, k = 3) 
dim(train.norm.df)
fundraisingKnn1 <- knn(train = train.df[, x], test = test.df[, x], 
                      cl = train.df[, y], k = 3) 

# WITHOUT NORMALIZATION
confusionMatrix(fundraisingKnn, as.factor(valid.norm.df[, y]))

# WITH NORMALIZATION
confusionMatrix(fundraisingKnnnorm, as.factor(valid.norm.df[, y]))

# WITHOUT NORMALIZATION for test data
confusionMatrix(fundraisingKnn1, as.factor(test.norm.df[, y]))

# WITH NORMALIZATION for test data
confusionMatrix(fundraisingKnnnorm1, as.factor(test.norm.df[, y]))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 5: Identifying Best k

library(caret)

# initialize a data frame with two columns: k, and accuracy.
fundraisingaccuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14)) #rep just repeats a value (0 in this case) 14 times. We are just initiating accuracy
fundraisingaccuracy.df

# compute knn for different k on validation.
for(i in 1:14) {
  #Use knn function with k=i and predict for valid dataset
 fundraisingknn.pred <- knn(train = train.norm.df[, x], test = valid.norm.df[, x], 
                          cl = as.factor(train.norm.df[, y]), k = i)
  fundraisingaccuracy.df[i, 2] <- confusionMatrix(fundraisingknn.pred, as.factor(valid.df[, y]))$overall[1] 
}

fundraisingaccuracy.df

fundraisingknn.pred <- knn(train = train.norm.df[, x], test = valid.norm.df[, x], 
                        cl = as.factor(train.norm.df[, y]), k = 3)
confusionMatrix(fundraisingknn.pred, as.factor(valid.df[, y]))

#k = 14

library(FNN)
# Normalized Dataframe
fundraisingKnnnorm2 <- knn(train = train.norm.df[, x], test = valid.norm.df[, x], 
                          cl = train.norm.df$TARGET_B, k = 14) 
dim(train.norm.df)
fundraisingKnn2 <- knn(train = train.df[, x], test = valid.df[, x], 
                      cl = train.df[, y], k = 14) 


# WITHOUT NORMALIZATION
confusionMatrix(fundraisingKnn2, as.factor(valid.norm.df[, y]))

# WITH NORMALIZATION
confusionMatrix(fundraisingKnnnorm2, as.factor(valid.norm.df[, y]))
