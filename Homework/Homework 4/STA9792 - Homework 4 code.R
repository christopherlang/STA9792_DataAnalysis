library(tidyverse)
library(neuralnet)
library(iterators)
library(snow)
library(doSNOW)
library(foreach)

training_data <- read_csv("Homework/Homework 4/HW04_Data_data_tab.csv")
validation_data <- read_csv("Homework/Homework 4/HW04_Data_prediction_tab.csv")

set.seed(12345)

# onehot_vector function defined ===============================================
# Function to convert a factor vector into a "one hot vector" data frame
onehot_vector <- function(y, value = 1, fill = 0) {
  classes <- unique(y)

  classes <- (
    data_frame(id = 1:length(y), classes = y,
               values = value) %>%
      spread(classes, values, fill = fill) %>%
      select(-id)
  )

  return(classes)
}

# Homework code ================================================================
# The neuralnet package needs to have "one hot vector" for classification
# Essentially, each class will need to be its own column
training_data <- (
  training_data %>%
    mutate(y = factor(y, seq(0, 0.9, 0.1), paste0('C', seq(0, 9, 1)),
                      ordered = T))
)

vector_classes <- training_data$y
hotvector_classes <- onehot_vector(vector_classes)

training_data <- (
  bind_cols(training_data, hotvector_classes) %>%
    select(-y)
)

# Split training data into two datasets:
# - learning dataset
# - testing dataset
#
# Ensure that both dataset has similar representation from the original classes
learn_prop <- 0.8125
split_indices <- sample(seq_len(nrow(training_data)))

split_dataset <- by(split_indices, vector_classes, function(x) {
  list(
    learning = x[1:(learn_prop * length(x))],
    testing = x[(learn_prop * length(x) + 1):length(x)]
  )
})

learning_dataset <- unlist(lapply(split_dataset, function(x) x$learning))
learning_dataset <- training_data[learning_dataset,]

testing_dataset <- unlist(lapply(split_dataset, function(x) x$testing))
testing_dataset <- training_data[testing_dataset,]

testing_class <- (
  testing_dataset %>%
    select(num_range('C', 0:9)) %>%
    mutate(id = 1:n()) %>%
    gather(classes, value, -id) %>%
    group_by(id) %>%
    filter(value > 0) %>%
    arrange(id) %>%
    ungroup() %>%
    select(-id, -value)
)

testing_class <- testing_class[['classes']]

testing_dataset <- (
  testing_dataset %>%
    select(num_range('x', 1:15))
)

rm(split_indices, split_dataset, learn_prop)
# For this homework, we will use a standard feedforward neural network, with
# one hidden layer
# For the model's hyperparameters, we will test the following:
#   - 15 input neurons, one for each predictor
#   - 10, 15, 20 hidden layer neurons (arbitrarily chosen)
#   - 10 output neurons, one for each number
#   - Logistic or tanh activation functions
#
# Since we already have a validation dataset, we will use all of training for
# generating models. For the tuning of the neural networ hyperparameters, we
# will use kfold cross-validation, specifically k = 10

model_formula <- (
  paste(paste0('C', 0:9, collapse = " + "),
        paste0('x', 1:15, collapse = " + "),
        sep = " ~ ")
)
model_formula <- as.formula(model_formula)

# Create grid of parameters for testing ========================================
# The parameters include
grid_params <- list(
  hidden = c(1, 2, 3, 4, 5, 10, 15, 20),
  act.fct = c('logistic', 'tanh'),
  threshold = c(0.01, 0.03, 0.05, 0.1)
)
grid_params <- expand.grid(grid_params)

# Configure parallel computation backend
cl <- makeCluster(7)  # I have 8 logical processors, using 7 to avoid slowdown
registerDoSNOW(cl)

# A progress bar always good to have
pb <- txtProgressBar(max = nrow(grid_params), style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

r <- foreach(
  pp = iter(grid_params, by = 'row'),
  .export = c('learning_dataset', 'testing_dataset', 'testing_class'),
  .packages = c('tidyverse', 'neuralnet'),
  .options.snow = opts) %dopar% {
    act_fun <- pp$act.fct
    nneurons <- pp$hidden
    thres <- pp$threshold

    model <- neuralnet(model_formula, data = learning_dataset,
                       hidden = nneurons,
                       act.fct = act_fun,
                       err.fct = 'sse',
                       linear.output = FALSE,
                       lifesign = 'minimal',
                       threshold = thres)

    test_predictions <- tryCatch({
      neuralnet::compute(model, testing_dataset)
    }, error = function(e) {
      NULL
    })

    result <- (
      data_frame(
        act_fun = act_fun,
        nneurons = nneurons,
        threshold = thres,
        total = length(testing_class)
      )
    )

    if (!is.null(test_predictions)) {
      # Select predicted class based only on max value of NN outputs
      test_predictions <- apply(test_predictions$net.result, 1,
                                function(x) which.max(x))
      test_predictions <- paste0('C', test_predictions - 1)

      result <- (
        result %>%
          mutate(accuracy = sum(test_predictions == testing_class)) %>%
          mutate(accuracy = accuracy / length(test_predictions))
      )

    } else {
      result <- (
        result %>%
          mutate(accuracy = NA)
      )
    }

    return(result)
  }

stopCluster(cl)

r <- (
  r %>%
    bind_rows() %>%
    arrange(desc(accuracy), nneurons, threshold)
)

# Several instances of same correct classification rate (accuracy)
# We select the best by accuracy:

nn_model <- slice(r, 1)

nn_model <- neuralnet(model_formula, data = training_data,
                      hidden = nn_model$nneurons,
                      act.fct = nn_model$act_fun,
                      err.fct = 'sse',
                      linear.output = FALSE,
                      lifesign = 'minimal',
                      threshold = nn_model$threshold)

final_result <- compute(nn_model, validation_data)
final_result <- apply(final_result$net.result, 1, which.max)
final_result <- final_result - 1

writeLines(as.character(final_result / 10), 'clipboard')
writeLines(as.character(final_result), 'clipboard')
