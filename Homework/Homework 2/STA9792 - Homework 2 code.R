library(tidyverse)

# Question 2.1
stock_data <- read_csv('Homework/Homework 2/stock_data.csv')
riskfree_return <- read_csv("Homework/Homework 2/date_riskfree_return.csv")
market_data <- read_csv('Homework/Homework 2/market_close.csv')

riskfree_return <- rename(riskfree_return, riskfree_rf = `Risk Free Rf`)

stock_data <- (
  stock_data %>%
    mutate(Date = as.Date(Date, format = '%m/%d/%Y')) %>%
    gather(symbol, close, -Date)
)

riskfree_return <- (
  riskfree_return %>%
    mutate(Date = as.Date(Date, format = '%m/%d/%Y'))
)

market_data <- (
  market_data %>%
    mutate(Date = as.Date(Date, format = '%m/%d/%Y'))
)

# This function takes original stock data and creates the necessary
# predictor X (the natural log of market index - risk free return)
# response Y (the natural log of stock return - risk free return)
# columns. This new output data.frame is used for linear regression modeling
model_data <- function(stock_data2, riskfree_return, market_data) {
  ticker_symbol <- unique(stock_data2$symbol)
  modelframe <- select(stock_data2, -symbol) %>% arrange(Date)

  modelframe <- (
    modelframe %>%
      mutate(ln_return = log(close / lag(close))) %>%
      left_join(riskfree_return, by = c('Date' = 'Date')) %>%
      mutate(ln_return_rf_stock = ln_return - riskfree_rf) %>%
      select(-close, -ln_return)
  )

  modelframe <- (
    modelframe %>%
      left_join(market_data, by = c('Date' = 'Date')) %>%
      mutate(ln_return = log(close / lag(close))) %>%
      mutate(ln_return_rf_market = ln_return - riskfree_rf) %>%
      select(-close, -ln_return, -riskfree_rf) %>%
      filter(!is.na(ln_return_rf_stock)) %>%
      mutate(stock_symbol = ticker_symbol) %>%
      select(Date, stock_symbol, everything())
  )
}

model_frames <- by(stock_data, list(stock_data$symbol), model_data,
                   riskfree_return = riskfree_return, market_data = market_data,
                   simplify = FALSE)

asset_capm <- lapply(model_frames, function(x) {
  model <- lm(ln_return_rf_stock ~ ln_return_rf_market, data = x)
  ticker <- unique(x$stock_symbol)

  beta_value <- coef(summary(model))[2, 1]
  beta_se <- coef(summary(model))[2, 2]

  t_value <- (beta_value - 1) / beta_se
  t_crit <- 1.645

  if (t_value < t_crit) {
    beta_is_significant <- TRUE
  } else {
    beta_is_significant <- FALSE
  }

  return(list(model = model, stock = ticker,
              significant = beta_is_significant,
              beta_value = beta_value, beta_se = beta_se))
})

final_result <- lapply(asset_capm, function(x) {
  r <- data_frame(stock = x$stock, beta = x$beta_value, beta_se = x$beta_se,
                  beta_095 = x$significant)

  return(r)
})

final_result <- bind_rows(final_result)

# Question 2.2
# Yes

fama_french <- read_csv("Homework/Homework 2/fama_french_factors.csv")
fama_french <- (
  fama_french %>%
    mutate(Date = as.Date(Date, format = "%m/%d/%Y")) %>%
    rename(mkt_rf = `(Mkt-RF)*`, smb = `SMB*`, hml = `HML*`)

)

model_data <- function(stock_data2, riskfree_return, market_data, fama_french) {
  ticker_symbol <- unique(stock_data2$symbol)
  modelframe <- select(stock_data2, -symbol) %>% arrange(Date)

  modelframe <- (
    modelframe %>%
      mutate(ln_return = log(close / lag(close))) %>%
      left_join(riskfree_return, by = c('Date' = 'Date')) %>%
      mutate(ln_return_rf_stock = ln_return - riskfree_rf) %>%
      select(-close, -ln_return)
  )

  modelframe <- (
    modelframe %>%
      left_join(market_data, by = c('Date' = 'Date')) %>%
      mutate(ln_return = log(close / lag(close))) %>%
      mutate(ln_return_rf_market = ln_return - riskfree_rf) %>%
      select(-close, -ln_return, -riskfree_rf) %>%
      filter(!is.na(ln_return_rf_stock)) %>%
      mutate(stock_symbol = ticker_symbol) %>%
      select(Date, stock_symbol, everything())
  )

  modelframe <- (
    modelframe %>%
      left_join(fama_french, by = c('Date' = 'Date'))
  )

  return(modelframe)
}

model_frames <- by(stock_data, list(stock_data$symbol), model_data,
                   riskfree_return = riskfree_return, market_data = market_data,
                   fama_french = fama_french, simplify = FALSE)

asset_famafrench <- lapply(model_frames, function(x) {
  model <- lm(ln_return_rf_stock ~ mkt_rf + smb + hml, data = x)
  ticker <- unique(x$stock_symbol)

  beta_value <- coef(summary(model))[2, 1]
  beta_se <- coef(summary(model))[2, 2]

  t_value <- (beta_value - 1) / beta_se
  t_crit <- 1.645

  if (t_value < t_crit) {
    beta_is_significant <- TRUE
  } else {
    beta_is_significant <- FALSE
  }

  return(list(model = model, stock = ticker,
              significant = beta_is_significant,
              beta_value = beta_value, beta_se = beta_se))
})


# Question 2.3
logit_data <- read_csv("Homework/Homework 2/logitmodel.csv")
logit_data <- mutate_all(logit_data, funs(as.numeric(.)))

logit_data <- (
  logit_data %>%
    mutate(new_prob = log(1 / (1 - Prob)))
)

model <- lm(new_prob ~ X1 + X2 + X3, data = logit_data)

predicted_prob <- predict(model, newdata = logit_data)

round(summary((logit_data$Prob - exp(predicted_prob) / (1 + exp(predicted_prob)))^2), 4)

