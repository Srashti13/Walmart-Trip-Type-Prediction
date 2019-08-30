# Walmart Project

# --- Libraries Required ---
library(reshape2)
library(plyr)
#install.packages("xgboost")
library(xgboost)
library(randomForest)
library(ModelMetrics)
library(caret)

# --- Set seed for reproducibility ---
set.seed(23456)

# --- Reading the data ---
walMrt_df <- read.csv("train.csv")

# Shape and variable description of the data
print(paste0("Number of rows: ", nrow(walMrt_df)))
print(paste0("Number of columns: ", ncol(walMrt_df)))

# --- Data fields ---
#TripType    - a categorical id representing the type of shopping trip the customer made. 
#              This is the ground truth that you are predicting. TripType_999 is an 
#              "other" category.
#VisitNumber - an id corresponding to a single trip by a single customer
#Weekday     - the weekday of the trip
#Upc         - the UPC number of the product purchased
#ScanCount   - the number of the given item that was purchased. A negative value indicates
#              a product return.
#DepartmentDescription - a high-level description of the item's department
#FinelineNumber - a more refined category for each of the products, created by Walmart

# data description
str(walMrt_df)

# --- Exploratory Data Analysis and Data Cleansing ---

# 1. Checking for Na values
apply(walMrt_df, 2, function(x) sum(is.na(x)))

# As the number of rows is 647054 while the max number of na rows in 4129, removing them
# would not have a significant impact on the dataset

# Removing the 'na' rows
walMrt_df <- walMrt_df[complete.cases(walMrt_df),]

# 2. Creating dummy variables for Weekday & department description.
walMrt_df2 <- dcast(walMrt_df, TripType+VisitNumber+Weekday+FinelineNumber~DepartmentDescription,
                    length, value.var = "ScanCount")

# "length", This is a function used for track number of transactions in each FinelineNumber, while
# we can also use "sum", as an alternate function to track the total number of item at each FinelineNumber
# but in "sum", there is an conflict on how to interpret return items (i.e. -ive values), as it subtracts
# the number of return item from overall purchase 
# Run & check eg. walMrt_df2[walMrt_df2$VisitNumber == 8,] & walMrt_df[walMrt_df$VisitNumber == 8,]

# creating dummy variables for Weekday (https://stackoverflow.com/questions/11952706/generate-a-dummy-variable)
make_dummies <- function(v, prefix = '') {
  s <- sort(unique(v))
  d <- outer(v, s, function(v, s) 1L * (v == s))
  colnames(d) <- paste0(prefix, s)
  d
}

# bind the dummies to the original dataframe
walMrt_df3 <- cbind(walMrt_df2, make_dummies(walMrt_df2$Weekday, prefix = ''))

# (to-do)drop the weekday column!
walMrt_df3 <- walMrt_df3[,!(names(walMrt_df3) %in% "Weekday")]

# eg (for checking)
#walMrt_df[walMrt_df$VisitNumber == 106,]

# 3. Re-Mapping the trip type to sequential numbers from 0 rather than randomly assigned numbers
map_triptype <- c("3"="0", '4'='1', '5'='2', '6'='3', '7'='4', '8'='5', '9'='6', 
                  '12'='7', '14'='8', '15'='9', '18'='10', '19'='11', '20'='12', 
                  '21'='13', '22'='14', '23'='15', '24'='16', '25'='17', '26'='18', 
                  '27'='19', '28'='20', '29'='21', '30'='22', '31'='23', '32'='24',
                  '33'='25', '34'='26', '35'='27', '36'='28', '37'='29', '38'='30',
                  '39'='31', '40'='32', '41'='33', '42'='34', '43'='35', '44'='36',
                  '999'='37')
walMrt_df3$TripType <- factor(walMrt_df3$TripType)
walMrt_df3$TripType <- revalue(walMrt_df3$TripType, map_triptype)

# --- Test and Train Dataset ---
idx <- sample(1:nrow(walMrt_df3),0.7*nrow(walMrt_df3))
test <- walMrt_df3[-idx,]
train <- walMrt_df3[idx,]


# --- Modeling ---

# -- XGBoost --
noOfClasses <- length(unique(walMrt_df3$TripType))
Target <- as.numeric(as.character(train$TripType))

# Create DMatrix object from train data set - Will be used for model generation.
trainMatrix <- xgb.DMatrix(data = data.matrix(train[!(names(train) %in% c("TripType", "VisitNumber"))]),
                           label = Target)

# Prepare parameter list - - Will be used for model generation.
# Add more parameters for fine tuning or for early stopping.
param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = noOfClasses)

# Initialize number of rounds and folds. Try with a bigger number initially, and perform cross-validation.
# This is required to identify 'Global-minimum' instead of getting stuck with 'Local-minimum'
cv.round <- 200
cv.nfold <- 5

### Model Generation
# Perform Cross-validation using the above params and objects
#xgbcv <- xgb.cv(param = param, data = trainMatrix,
#                label = Target, nrounds = cv.round, 
#                nfold = cv.nfold)

# Plot to visualize how the cross-validation is performing
# plot(xgbcv$test.mlogloss.mean, type='l')

# Determine 'Global-minimum' number of rounds required for the model
#nround <- which(xgbcv$test.mlogloss.mean == min(xgbcv$test.mlogloss.mean) )

# Develop a model using XGBoost and the above params / objects 
xgb_model <- xgboost(param = param, data = trainMatrix, label = Target, 
                     max_depth = 15, eta = 0.05, nrounds = 100)

# --- Prediction ---
# Create Dense matrix  from test data set - Will be used for prediction.
testMatrix <- as.matrix(test[!(names(train) %in% c("TripType", "VisitNumber"))])

# Predict the value
ypred <- predict(xgb_model, testMatrix)

# Convert predicted values into Matrix as stated in sample-submission
predMatrix <- data.frame(matrix(ypred, byrow = TRUE, ncol = noOfClasses))

### Output
# Create column header for Output file
colnames(predMatrix) <- paste("TripType_", 0:37, sep="")

# Combine column header and predicted values as data frame
res <- data.frame(VisitNumber = test[, 2], predMatrix)

# Calculating the logloss for multiclassification
mlogLoss(test$TripType, predMatrix)
#2.253352

#predicted triptype
#res$pred <- unlist(lapply(colnames(res[,-1])[apply(res[,-1],1, which.max)], 
#                          function(x) as.numeric(strsplit(x,"_")[[1]][2])))

# Perform aggregation on Visit number by taking 'average'
#result <- aggregate(. ~ VisitNumber, data = res, FUN = mean)


# --- Random Forest ---

### Model Generation

rf_model <- randomForest(train[!(names(train) %in% c("TripType", "VisitNumber"))],
                         y = train$TripType, mtry = 60, ntree = 225, maxnodes = 40)


# alternative way :-
# Prepare Train Control 
# Do performance tuning by adjusting 'number of folds', CV Repeats or by adding more params
#trcontrol <- trainControl(method = "repeatedcv", number = 4, repeats = 2,
#                          verboseIter = FALSE, returnResamp = "all", classProbs = TRUE)

# Determine Number of Classes to be predicted - Will be used for model generation.
#RF_MTRY = length(unique(train$TripType))

# Set no of Trees for Random Forest
# Do performance tuning by adjusting number of trees
#RF_TREES = 225

# Prepare Tune Grid object
#tGrid <- expand.grid(mtry = RF_MTRY)

# Develop a model using Random Forest and the above params / objects 
#rf_model2 <- train(x = train[!(names(train) %in% c("TripType", "VisitNumber"))],
#                   y = train$TripType, method = "rf",  trControl = trcontrol,
#                   tuneGrid = tGrid, metric = "Accuracy", ntree = RF_TREES)

# Prediction
# Predict the value
ypred2 <- predict(rf_model, test[!(names(train) %in% c("TripType", "VisitNumber"))], type="prob")

# Convert predicted values into Matrix as stated in sample-submission
predMatrix2 <- data.frame(matrix(ypred2, byrow = TRUE, ncol = noOfClasses))

# set column names
colnames(predMatrix2) <- paste("TripType_", 0:37, sep="")

# Calculating the logloss for multiclassification
mlogLoss(test$TripType, predMatrix2)

ggplot
# Visualizations
x <- data.frame(table(walMrt_df$DepartmentDescription))
x <- x[order(-x$Freq),]
x <- x[1:30,]
rownames(x) <- 1:nrow(x)

ggplot(x, aes(x = reorder(Var1, Freq), y = Freq)) + geom_bar(stat = "identity") + 
  coord_flip() + ggtitle(" Top 30 departments Products were purchased in each trip") +
                           xlab("Departments") + ylab("Counts")


y <- unique(walMrt_df[,c('VisitNumber','Weekday')])
y$Weekday <- factor(y$Weekday, levels = c("Sunday", "Monday", "Tuesday", "Wednesday",
                                          "Thursday", "Friday", "Saturday"))
ggplot(y, aes(Weekday, fill= Weekday)) + geom_bar() + coord_flip() +
  ggtitle("Number of Unique trips made each day of the week")

