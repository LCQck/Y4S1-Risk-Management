spy = read.csv('SPY.csv', header = TRUE)
gld = read.csv('GLD.csv', header = TRUE)
tlt = read.csv('TLT.csv', header = TRUE)

install.packages("rugarch")
install.packages("rmgarch")
install.packages("copula")

library(rugarch)
library(rmgarch)
library(copula)

spy_log_returns <- data.frame(Date = spy$Date[-1], Log_Returns = diff(log(spy$Close)))
gld_log_returns <- data.frame(Date = gld$Date[-1], Log_Returns = diff(log(gld$Close)))
tlt_log_returns <- data.frame(Date = tlt$Date[-1], Log_Returns = diff(log(tlt$Close)))


# Fit a univariate GARCH model to each asset's returns
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
                   distribution.model = "norm")

spy_garch <- ugarchfit(spec, spy$log_returns)
gld_garch <- ugarchfit(spec, gld$log_returns)
tlt_garch <- ugarchfit(spec, tlt$log_returns)

# Extract the standardized residuals
spy_resid <- residuals(spy_garch, standardize = TRUE)
gld_resid <- residuals(gld_garch, standardize = TRUE)
tlt_resid <- residuals(tlt_garch, standardize = TRUE)