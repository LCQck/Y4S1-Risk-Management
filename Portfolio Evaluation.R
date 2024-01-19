# Data preprocessing and portfolio construction 
# Read data
gld = read.csv('GLD.csv', header = TRUE)
sp500 = read.csv('SPY.csv', header = TRUE)
tll = read.csv('TLT.csv', header = TRUE)

#Calculate the daily log-return
r_gld = cbind(gld$Date[-1], diff(log(gld$Close)))
r_sp500 = cbind(sp500$Date[-1], diff(log(sp500$Close)))
r_tll = cbind(tll$Date[-1], diff(log(tll$Close)))
# Merge the same trading days
df = merge(r_gld, r_sp500, by='V1', all=FALSE)
df = merge(df, r_tll, by='V1', all=FALSE)
colnames(df) = c('Date', 'r_gld', 'r_sp500', 'r_tll')
df = na.omit(df)# Remove the null value
# Divide the data set 20070103-20171228 is a fitting data set, and 20171228-20211230 is a set of prediction VaR and evaluation of VaR.
timestamp = df[,1]
loc = which(timestamp==20171228)
loc
train_df = df[1:loc,]
test_df = df[(loc+1):dim(df)[1],]
# Get the daily return of two sets
train_return = train_df[,2:4]
test_return = test_df[,2:4]
# Build a portfolio weight
N = 3
# equal weight portfolio
w = rep(1,N)/N
View(w)


#A suitable GARCH model is fitted to the return rate on the train.
#
install.packages('rugarch')
install.packages('forecast')
install.packages('e1071')
library('rugarch')
library('xts')
library('forecast')
library('e1071')
#Convert to date data
train_time = as.Date(as.character(train_df[,1]), '%Y%m%d')
test_time = as.Date(as.character(test_df[,1]), '%Y%m%d')
# ------ Overall statistical description
# The overall sample average, standard deviation, bias and peak of the three stocks
#（Normal distribution bias = 0; peak = 3）
c(mean(df[,2]), sd(df[,2]), skewness(df[,2]), kurtosis(df[,2]))
c(mean(df[,3]), sd(df[,3]), skewness(df[,3]), kurtosis(df[,3]))
c(mean(df[,4]), sd(df[,4]), skewness(df[,4]), kurtosis(df[,4]))
# Normality, if the scattering point does not fall on the straight line, it lacks normality, so T is used when fitting the marginal distribution below.distribution(t_G)
qqnorm(df[,2], main = 'gld daily log return')
qqline(df[,2])
qqnorm(df[,3], main = 'sp500 daily log return')
qqline(df[,3])
qqnorm(df[,4], main = 'tll daily log return')
qqline(df[,4])
#Self-correlation graph (used to judge MA) and partial self-correlation graph (used to judge AR) determine whether it is related to past return and past shock.
#PACF judges that the first line of AR exceeds the dotted line, and there must be AR.
#ACFDon't worry about it at 0 degrees. It's equal to 1.
acf(df[,2], main = 'ACF of gld daily log return')
acf(df[,3], main = 'ACF of sp500 daily log return')
acf(df[,4], main = 'ACF of tll daily log return')
pacf(df[,2], main = 'PACF of gld daily log return')
pacf(df[,3], main = 'PACF of sp500 daily log return')
pacf(df[,4], main = 'PACF of tll daily log return')
# ------Modelling
# for the first stock index return
#First, build the Mean Equation of the GARCH model: ARMA(p,q) model
# Use the auto.arima function to automatically search for the most suitable p and q values
fit_meanEq_1 = auto.arima(train_return[,1])
summary(fit_meanEq_1)#arma=（0，0） mean eq :rt =et
# Test mean equation的adequency
Box.test(residuals(fit_meanEq_1), lag = 12, type = 'Ljung-Box')        # The residuals should not be relevant. The p-value should be greater than 0.05
Box.test(residuals(fit_meanEq_1)^2, lag = 12, type = 'Ljung-Box')       # The residual square should be correlated (the existence of ARCH effect will be modelled with the GARCH model) p-value should be less than 0.05
p = fit_meanEq_1$arma[1]
q = fit_meanEq_1$arma[2]
p
q

# The Mean Equation based on ARMA(p,q) above continues to build the Variance equation: GARCH(m,n) model, in order to eliminate the residual ARCH effect
# Use the AIC minimum grid to search for the appropriate m and n
max_lag = 5
AIC_metrics = matrix(rep(0, max_lag^2), max_lag, max_lag)
for (i in 1:max_lag){
  for (j in 1:max_lag){
    GARCHij = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(i,j)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
    GARCHij_fit = ugarchfit(GARCHij, train_return[,1])
    AIC_metrics[i,j] = infocriteria(GARCHij_fit)['Akaike',]
  }
}
m = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[1]
n = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[2]

AIC_metrics
m
n
# Build the best mean equation and variety equation based on the above p, q and m, n
GARCH1 = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(m,n)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
GARCH1_fit = ugarchfit(GARCH1, train_return[,1])
GARCH1_fit
# return fitted(This is the prediction in the sample.)
return_fitted_1 = fitted(GARCH1_fit)
# conditional volatility fitted
ConVolatility_1 = sigma(GARCH1_fit)
# residuals
residuals_1 = residuals(GARCH1_fit)
# standardized residuals
standRes_1 = residuals_1/ConVolatility_1
# model assessment
# plot the conditional volatility
Tvol1 = xts(ConVolatility_1, train_time)
plot(Tvol1)
plot(Tvol1, format.labels = '%d-%m-%Y')
# Test the adequency (standard residual test) after being fitted by GARCH
Box.test(standRes_1, lag = 12, type = 'Ljung-Box')          # The standard residual should not have a correlation p-value should be greater than 0.05
Box.test(standRes_1^2, lag = 12, type = 'Ljung-Box')        # The standard residual square should not have a correlation p-value greater than 0.05 (ARCH effect eliminated)
# Predict the prediction set
garchroll_1 = ugarchroll(GARCH1, data = df[,2] , n.start = loc, refit.window = 'moving',  refit.every = 10)
preds_1 = as.data.frame(garchroll_1)#Turn the prediction into a data framework.
preds_1_mu = as.xts(preds_1$Mu, test_time)
preds_1_sigma = as.xts(preds_1$Sigma, test_time)
# The following are the repeated steps to carry out the same modelling process for the rest of the stocks.
# for the second stock index return
# First build the Mean Equation of the GARCH model: ARMA(p,q) model
# Use the auto.arima function to automatically search for the most suitable p and q values.
fit_meanEq_2 = auto.arima(train_return[,2])
summary(fit_meanEq_2)
p = fit_meanEq_2$arma[1]
q = fit_meanEq_2$arma[2]
p
q

Box.test(residuals(fit_meanEq_2), lag = 12, type = 'Ljung-Box')          
Box.test(residuals(fit_meanEq_2)^2, lag = 12, type = 'Ljung-Box')       

max_lag = 5
AIC_metrics = matrix(rep(0, max_lag^2), max_lag, max_lag)
for (i in 1:max_lag){
  for (j in 1:max_lag){
    GARCHij = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(i,j)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
    GARCHij_fit = ugarchfit(GARCHij, train_return[,2])
    AIC_metrics[i,j] = infocriteria(GARCHij_fit)['Akaike',]
  }
}
m = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[1]
n = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[2]
m
n
AIC

GARCH2 = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(m,n)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
GARCH2_fit = ugarchfit(GARCH2, train_return[,2])
GARCH2_fit

# return fitted
return_fitted_2 = fitted(GARCH2_fit)
# conditional volatility fitted
ConVolatility_2 = sigma(GARCH2_fit)
# residuals
residuals_2 = residuals(GARCH2_fit)
# standardized residuals
standRes_2 = residuals_2/ConVolatility_2

# model assessment
# plot the conditional volatility
Tvol2 = xts(ConVolatility_2, train_time)
plot(Tvol2)
plot(Tvol2, format.labels = '%d-%m-%Y')

Box.test(standRes_2, lag = 12, type = 'Ljung-Box')         # The standard residual should not have correlation p-value should be greater than 0.05
Box.test(standRes_2^2, lag = 12, type = 'Ljung-Box')       # The standard residual square should not have correlation p-value should be greater than 0.05 (ARCH effect eliminated)

# Predict the prediction setgarchroll_2 = ugarchroll(GARCH2, data = df[,3] , n.start = loc, refit.window = 'moving',  refit.every = 10)
preds_2 = as.data.frame(garchroll_2)
preds_2_mu = as.xts(preds_2$Mu, test_time)
preds_2_sigma = as.xts(preds_2$Sigma, test_time)



# for the third stock index return

fit_meanEq_3 = auto.arima(train_return[,3])
summary(fit_meanEq_3)
p = fit_meanEq_3$arma[1]
q = fit_meanEq_3$arma[2]
p
q


Box.test(residuals(fit_meanEq_3), lag = 12, type = 'Ljung-Box')          
Box.test(residuals(fit_meanEq_3)^2, lag = 12, type = 'Ljung-Box')       


max_lag = 5
AIC_metrics = matrix(rep(0, max_lag^2), max_lag, max_lag)
for (i in 1:max_lag){
  for (j in 1:max_lag){
    GARCHij = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(i,j)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
    GARCHij_fit = ugarchfit(GARCHij, train_return[,3])
    AIC_metrics[i,j] = infocriteria(GARCHij_fit)['Akaike',]
  }
}
m = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[1]
n = which(AIC_metrics == min(AIC_metrics), arr.ind = TRUE)[2]
m
n
AIC


GARCH3 = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(m,n)), mean.model = list(armaOrder = c(p,q), include.mean = FALSE), distribution.model = 'std')
GARCH3_fit = ugarchfit(GARCH3, train_return[,3])
GARCH3_fit

# return fitted
return_fitted_3 = fitted(GARCH3_fit)
# conditional volatility fitted
ConVolatility_3 = sigma(GARCH3_fit)
# residuals
residuals_3 = residuals(GARCH3_fit)
# standardized residuals
standRes_3 = residuals_3/ConVolatility_3

# model assessment
# plot the conditional volatility
Tvol3 = xts(ConVolatility_3, train_time)
plot(Tvol3)
plot(Tvol3, format.labels = '%d-%m-%Y')
# Test the adequency (standard residual test) after GARCH fitting
Box.test(standRes_3, lag = 12, type = 'Ljung-Box')          # The standard residual should not have correlation p-value should be greater than 0.05
Box.test(standRes_3^2, lag = 12, type = 'Ljung-Box')     # The standard residual square should not have correlation p-value should be greater than 0.05 (ARCH effect eliminated)

# Predict the prediction set
garchroll_3 = ugarchroll(GARCH3, data = df[,4] , n.start = loc, refit.window = 'moving',  refit.every = 10)
preds_3 = as.data.frame(garchroll_3)
preds_3_mu = as.xts(preds_3$Mu, test_time)
preds_3_sigma = as.xts(preds_3$Sigma, test_time)


#Because the non-normal distribution is judged at the beginning, it is more appropriate to use std or sstd instead of norm in garch.
#-------------------------------------------------------
# Fit the copula function
#
install.packages('fitdistrplus')
install.packages('copula')
library('fitdistrplus')
library('copula')

standRes_1 = as.vector(standRes_1)
standRes_2 = as.vector(standRes_2)
standRes_3 = as.vector(standRes_3)

#The code Generalised t-distributed p, d, q functions appear in the courseware
dt_G = function(x, mean, sd, nu){#density
  dt((x-mean)/sd,nu)/sd
}
pt_G = function(q, mean, sd, nu){#CDF
  pt((q-mean)/sd,nu)
}
qt_G = function(x, mean, sd, nu){#quantile
  qt(x,nu)*sd+mean
}

# The standard residuals fitted by the above three GARCH models are fitted to their marginal distribution.
fit_r1 = fitdist(standRes_1, 't_G', start = list(mean = mean(standRes_1), sd = sd(standRes_1), nu = 5)) 
summary(ft_r1)
fit_r2 = fitdist(standRes_2, 't_G', start = list(mean = mean(standRes_2), sd = sd(standRes_2), nu = 5)) 
summary(ft_r2)
fit_r3 = fitdist(standRes_3, 't_G', start = list(mean = mean(standRes_3), sd = sd(standRes_3), nu = 5)) 
summary(ft_r3)

# Build a normal copula
u = matrix(nrow = length(standRes_1), ncol = N)
u[,1] = pt_G(standRes_1, mean = as.list(fit_r1$estimate)$mean, sd = as.list(fit_r1$estimate)$sd, nu = as.list(fit_r1$estimate)$nu)   
u[,2] = pt_G(standRes_2, mean = as.list(fit_r2$estimate)$mean, sd = as.list(fit_r2$estimate)$sd, nu = as.list(fit_r2$estimate)$nu) 
u[,3] = pt_G(standRes_3, mean = as.list(fit_r3$estimate)$mean, sd = as.list(fit_r3$estimate)$sd, nu = as.list(fit_r3$estimate)$nu) 
norm.cop = normalCopula(dim = N, dispstr = 'un')
n.cop = fitCopula(norm.cop, u, method = 'ml')
coef(n.cop)

#Obtain the parameters of three marginal distributions
mean_r1 = as.list(fit_r1$estimate)$mean
sd_r1 = as.list(fit_r1$estimate)$sd
nu_r1 = as.list(fit_r1$estimate)$nu

mean_r2 = as.list(fit_r2$estimate)$mean
sd_r2 = as.list(fit_r2$estimate)$sd
nu_r2 = as.list(fit_r2$estimate)$nu

mean_r3 = as.list(fit_r3$estimate)$mean
sd_r3 = as.list(fit_r3$estimate)$sd
nu_r3 = as.list(fit_r3$estimate)$nu





#-------------------------------------------------------
#
# Calculate VaR
#

# create the correlation matrix using the fitted normal copula
cor = matrix(data=c(1,coef(n.cop)[1],coef(n.cop)[2],coef(n.cop)[1],1,coef(n.cop)[3],coef(n.cop)[2],coef(n.cop)[3],1),nrow=3,ncol=3)
cor
# Monte Carlo Simulation
L = t(chol(cor))#Decompose cor
set.seed(1234)#Set a random number of seeds
M = 10000
T = length(test_time)
Sim_R1=matrix(nrow=M,ncol=T)
Sim_R2=matrix(nrow=M,ncol=T)
Sim_R3=matrix(nrow=M,ncol=T)
for (i in 1:M){
  z=rnorm(N*T)
  z=matrix(z,N,T)
  z_tilde=L%*%z
  R1=q_tG(pnorm(z_tilde[1,]),mean=mean_r1, sd=sd_r1, nu=nu_r1)
  R2=q_tG(pnorm(z_tilde[2,]),mean=mean_r2, sd=sd_r2, nu=nu_r2)
  R3=q_tG(pnorm(z_tilde[3,]),mean=mean_r3, sd=sd_r3, nu=nu_r3)
  Sim_R1[i,] = R1
  Sim_R2[i,] = R2
  Sim_R3[i,] = R3
}
# Fusion simulation value (simulated is standardlised residuals, VaR is the calculation of return) and prediction value
return_reintroduce_the_heteroscedasticity = function(pred_model, sigma_model, Simulated_returns){
  #sigma_matrix = coredata(sigma_model)
  diagonal_sigma = diag(as.numeric(sigma_model), nrow = length(sigma_model))
  simulated_residuals = diagonal_sigma%*%t(Simulated_returns)
  simulated_log_returns = simulated_residuals + matrix(rep(as.numeric(pred_model), ncol(simulated_residuals)), ncol = ncol(simulated_residuals))
  simulated_log_returns
}

return_1 = return_reintroduce_the_heteroscedasticity(preds_1_mu, preds_1_sigma, Sim_R1)
return_2 = return_reintroduce_the_heteroscedasticity(preds_2_mu, preds_2_sigma, Sim_R2)
return_3 = return_reintroduce_the_heteroscedasticity(preds_3_mu, preds_3_sigma, Sim_R3)

# Calculate the simulation value of the log return of the portfolio according to the selected portfolio
portfolio_return = w[1]*return_1+w[2]*return_2+w[3]*return_3
alpha = 0.01 # confidence level
VaR_p = rep(NA, T)
for (j in 1:T){
  # Sort the M analogue values of each day
  portfolio_return_sorted = sort(portfolio_return[j,])
# Take the smallest first a value (eg. The smallest 1%, at this time, is based on gain distribution, VaR is negative)
  VaR_p[j] = portfolio_return_sorted[floor(M*alpha)]
}
# Draw daily VaR
VaR_p = as.xts(VaR_p, test_time)
plot(VaR_p)
#-------------------------------------------------------
#
# Back-Test
# H0:VaR is suitable
# H1:VaR is not suitable
return_pf = w[1]*test_return[,1]+w[2]*test_return[,2]+w[3]*test_return[,3]
VaRTest(alpha, return_pf, VaR_p)


#-------------------------------------------------------
#
# historical simulation
#

VaR_p = rep(NA, T)
windowSize = 600
for (i in loc:dim(df)[1]-1){
  startloc = i-windowSize+1
  endloc = i
  selected_df = df[startloc:endloc,]
  gain = w[1]*selected_df[,2] + w[2]*selected_df[,3] + w[3]*selected_df[,4]
  VaR_p[i-loc+1] = sort(gain)[floor(windowSize*alpha)]
}

#Draw daily VaR
VaR_p = as.xts(VaR_p, test_time)
plot(VaR_p)

# Back Test
VaRTest(alpha, return_pf, VaR_p)
backTest(alpha, return_pf, VaR_p)


#-------------------------------------------------------
#
# Variance-Covariance Method
#

VaR_p = rep(NA, T)
windowSize = 600
for (i in loc:dim(df)[1]-1){
  startloc = i-windowSize+1
  endloc = i
  selected_df = df[startloc:endloc,]
  varcov = matrix(as.numeric(cov(selected_df[,2:4])), N, N)
  weight = as.vector(as.numeric(w))
  port_sd = sqrt(t(weight)%*%varcov%*%weight)
  VaR_p[i-loc+1] = qnorm(alpha)*port_sd
}

# Draw daily VaR
VaR_p = as.xts(VaR_p, test_time)
plot(VaR_p)

# Back Test
VaRTest(alpha, return_pf, VaR_p)
backTest(alpha, return_pf, VaR_p)

