#!/usr/bin/R

## see
# https://www.statmethods.net/stats/regression.html
# http://www.itl.nist.gov/div898/handbook/eda/section3/eda366.htm
# https://cran.r-project.org/web/packages/LogicReg/LogicReg.pdf


## LOAD the data
#wifi <- read.table("indoor_raw.txt",sep="\t", header=TRUE)
#wifi <- read.table("indoor_fix.txt",sep="\t", header=TRUE)
#plot(wifi, main=TITLE)

wifi <- read.table("indoor-logdist.txt",sep="\t", header=TRUE)
TITLE="Distância (m) vs RSSI (mW)"
#plot(wifi$dist,wifi$dB, main=TITLE)

x=wifi$dist
y_mW=wifi$mW
#y_dB=wifi$dB
y_dB=10*log10(y_mW)
y=y_mW

plot(x,y_dB, main=TITLE)

## INSTALL fitdistrplus
# echo '
#     install.packages("fitdistrplus")
#     install.packages("FAdist")
#     install.packages("MASS")
#     install.packages("bbmle")
# ' | sudo -i R --no-save --interactive

require("fitdistrplus")
require("FAdist")
# require("MASS")
# require("bbmle")

model_llog3 = fitdist(data=y_dB, dllog3, start=list(0.03483763, 5.23146324, -99.06433554))

model_logis = fitdist(data=-y_dB, distr="logis")
model_lnorm3 <- fitdist(data=-y_dB, dlnorm3, start=list(shape = 0.09152264, scale = 4.95512825, thres = -56.07357569))
model_gamma = fitdist(data=-y_dB, distr="gamma")

models=list(model_logis, model_lnorm3, model_gamma)
names=c("Logistic","3Par-LogNomal","Gamma")
gofstat(models,fitnames=names)
# Goodness-of-fit statistics
#                               Logistic 3Par-LogNomal     Gamma
# Kolmogorov-Smirnov statistic 0.1517015     0.1632569 0.1693002
# Cramer-von Mises statistic   0.3470420     0.9265190 0.9363562
# Anderson-Darling statistic   2.9566927     5.3358393 5.4342411

# Goodness-of-fit criteria
#                                Logistic 3Par-LogNomal    Gamma
# Akaike's Information Criterion 775.5513      802.5164 803.6166
# Bayesian Information Criterion 780.7616      810.3319 808.8270

plot(model_logis)
summary(model_logis)
# Fitting of the distribution ' logis ' by maximum likelihood 
# Parameters : 
#           estimate Std. Error
# location 88.185128   1.082976
# scale     6.282901   0.532610
# Loglikelihood:  -385.7757   AIC:  775.5513   BIC:  780.7616 
# Correlation matrix:
#            location      scale
# location  1.0000000 -0.1173162
# scale    -0.1173162  1.0000000

plot(model_lnorm3)
summary(model_lnorm3)
# Fitting of the distribution ' lnorm3 ' by maximum likelihood 
# Parameters : 
#           estimate  Std. Error
# shape   0.09121894  0.02849053
# scale   4.95529405  0.28055540
# thres -56.21239176 39.58281823
# Loglikelihood:  -398.2582   AIC:  802.5164   BIC:  810.3319 
# Correlation matrix:
#            shape      scale      thres
# shape  1.0000000 -0.9737957  0.9742179
# scale -0.9737957  1.0000000 -0.9994712
# thres  0.9742179 -0.9994712  1.0000000

plot(model_gamma)
summary(model_gamma)
# Fitting of the distribution ' gamma ' by maximum likelihood 
# Parameters : 
#         estimate Std. Error
# shape 42.2596118 5.95159063
# rate   0.4891678 0.06930072
# Loglikelihood:  -399.8083   AIC:  803.6166   BIC:  808.827 
# Correlation matrix:
#           shape      rate
# shape 1.0000000 0.9940876
# rate  0.9940876 1.0000000

plot(model_llog3)
summary(model_llog3)
# Fitting of the distribution ' llog3 ' by maximum likelihood 
# Parameters : 
#      estimate Std. Error
# 1   0.6238972 0.08765614
# 2   2.1584165 0.13734617
# 3 -99.4197743 0.51712110
# Loglikelihood:  -360.3787   AIC:  726.7573   BIC:  734.5728 
# Correlation matrix:
#            [,1]       [,2]       [,3]
# [1,]  1.0000000 -0.5121759  0.8076758
# [2,] -0.5121759  1.0000000 -0.6052262
# [3,]  0.8076758 -0.6052262  1.0000000

## Plot
m1 <- model_logis
m2 <- model_lnorm3
m3 <- model_gamma
#m3 <- model_llog3
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
cdfcomp(list(m1, m2, m3),  legendtext=c(m1$distname,m2$distname,m3$distname), xlogscale=FALSE)
denscomp(list(m1, m2, m3), legendtext=c(m1$distname,m2$distname,m3$distname) )
qqcomp(list(m1, m2, m3),   legendtext=c(m1$distname,m2$distname,m3$distname), xlogscale=FALSE, ylogscale=FALSE)
ppcomp(list(m1, m2, m3),   legendtext=c(m1$distname,m2$distname,m3$distname))







fitlm_best = function(x,y) {
	y=y_dB

	## Compare models
	fit1 <- lm(y~x)
	fit2 <- lm(y~poly(x,3))
	fit3 <- lm(y~log(x))
	fit4 <- lm(y~poly(log(x),3))

	layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
		plot(y~x, main="y~x")
		x2=seq(from=min(x),to=max(x),length.out=length(x))
		#y2=predict(fit1,newdata=list(x2),interval="confidence")
		y2=predict(fit1,newdata=list(x2))
		matlines(x2,y2,lwd=2)

		plot(y~x, main="y~poly(x,3)")
		x2=seq(from=min(x),to=max(x),length.out=length(x))
		#y2=predict(fit2,newdata=list(x2),interval="confidence")
		y2=predict(fit2,newdata=list(x2))
		matlines(x2,y2,lwd=2)

		plot(y~x, main="y~log(x)")
		x2=seq(from=min(x),to=max(x),length.out=length(x))
		#y2=predict(fit3,newdata=list(x2),interval="confidence")
		y2=predict(fit3,newdata=list(x2))
		matlines(x2,y2,lwd=2)

		plot(y~x,main="y~poly(log(x),3)")
		x2=seq(from=min(x),to=max(x),length.out=length(x))
		#y2=predict(fit4,newdata=list(x2),interval="confidence")
		y2=predict(fit4,newdata=list(x2))
		matlines(x2,y2,lwd=2)

	anova(fit1,fit2,fit3,fit4)

	## K-fold cross-validation
	#library(DAAG)
	#cv.lm(df=y, fit, m=3) # 3 fold cross-validation
}
# fitlm_best(x,y_mW)


fitdist_best = function(x,y) {
	## INSTALL fitdistrplus
	# echo 'install.packages("fitdistrplus")' | sudo -i R --no-save --interactive

	require("fitdistrplus")
	y=y_mW

	fit1 <- fitdist(data=y_mW, distr="lnorm")
	#summary(10*log10(rlnorm(n=1000, meanlog = -19.892263, sdlog = 2.770204)))
	
	fit2 <- fitdist(data=y_mW, distr="gamma")	
	#summary(10*log10(rgamma(n=1000, shape = 1.218814e-01, rate = 8.049905e+04)))
	
	fit3 <- fitdist(data=y_mW, distr="exp")
	summary(10*log10(rexp(n=1000, rate = 660834.7)))

	# fit3 <- fitdist(data=y_mW, distr="beta")
	# #summary(10*log10(rbeta(n=1000, shape1 =  1.218682e-01, shape2 = 8.049035e+04)))

	# fit4_dB <- fitdist(data=y_dB, distr="logis")
	# #summary(rlogis(n=1000, location = -88.153468, scale = 6.311918))
	# fit4 <- fitdist(data=y_mW, distr="logis", start=list(location=10^-88.153468))
	# #summary(10*log10(rlogis(n=1000, location = 6.052951e-07)))
	# fit4 <- fitdist(data=y_mW, distr="logis",start=(location=median(y_mW)))
	# #summary(10*log10(rlogis(n=1000, location = median(y_mW))))

	gofstat(list(fit1, fit2, fit3), fitnames=c(fit1$distname,fit2$distname,fit3$distname))
	summary(fit1)
	summary(fit2)
	summary(fit3)

	layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
	cdfcomp(list(fit1, fit2, fit3),  legendtext=c(fit1$distname,fit2$distname,fit3$distname), xlogscale=TRUE)
	denscomp(list(fit1, fit2, fit3), legendtext=c(fit1$distname,fit2$distname,fit3$distname), xlim=c(min(y_mW),median(y_mW)))
	qqcomp(list(fit1, fit2, fit3),   legendtext=c(fit1$distname,fit2$distname,fit3$distname), xlogscale=TRUE, ylogscale=TRUE)
	ppcomp(list(fit1, fit2, fit3),   legendtext=c(fit1$distname,fit2$distname,fit3$distname))

# > gofstat(list(fit1, fit2, fit3), fitnames=c(fit1$distname,fit2$distname,fit3$distname))
# Goodness-of-fit statistics
#                                  lnorm      gamma        exp
# Kolmogorov-Smirnov statistic 0.1473064  0.3412793  0.8429958
# Cramer-von Mises statistic   0.7080888  4.4885754 26.8320353
# Anderson-Darling statistic   4.2319595 21.6211852        Inf

# Goodness-of-fit criteria
#                                    lnorm     gamma       exp
# Akaike's Information Criterion -3486.881 -3331.364 -2478.252
# Bayesian Information Criterion -3481.670 -3326.154 -2475.647
# > summary(fit1)
# Fitting of the distribution ' lnorm ' by maximum likelihood 
# Parameters : 
#           estimate Std. Error
# meanlog -19.892263  0.2770204
# sdlog     2.770204  0.1958829
# Loglikelihood:  1745.44   AIC:  -3486.881   BIC:  -3481.67 
# Correlation matrix:
#               meanlog         sdlog
# meanlog  1.000000e+00 -3.084526e-09
# sdlog   -3.084526e-09  1.000000e+00

# > summary(fit2)
# Fitting of the distribution ' gamma ' by maximum likelihood 
# Parameters : 
#           estimate Std. Error
# shape 1.218814e-01          0
# rate  8.049905e+04        NaN
# Loglikelihood:  1667.682   AIC:  -3331.364   BIC:  -3326.154 
# Correlation matrix:
#       shape rate
# shape     1  NaN
# rate    NaN    1

# > summary(fit3)
# Fitting of the distribution ' exp ' by maximum likelihood 
# Parameters : 
#      estimate Std. Error
# rate 660834.7   2965.821
# Loglikelihood:  1240.126   AIC:  -2478.252   BIC:  -2475.647 


	# (4) defining your own distribution functions, here for the Gumbel distribution
	# for other distributions, see the CRAN task view 
	# dedicated to probability distributions
	#
	# dgumbel <- function(x, a, b) 1/b*exp((a-x)/b)*exp(-exp((a-x)/b))
	# pgumbel <- function(q, a, b) exp(-exp((a-q)/b)) #CDF!
	# qgumbel <- function(p, a, b) a-b*log(-log(p))	  #Median?

	# fitgumbel <- fitdist(data=y, distr="gumbel", start=list(a=10, b=10))

	#logdistance
	#PL0 = -69 dB; P0 = 10 m; Coef=3
	#logdist = PL0 - (10*C*LOG10(x/P0))
	# dlogdist <- function(x, a, b, c) a-(10*c*log10(x/b)) 
	# plogdist <- function(q, a, b, c) 1/(1+(1+((c*(q-a))/b)^(-1/c)))
	# qlogdist <- function(p, a, b, c) ( 1+(c*(p-a)/b)^(-(1/c+1)) ) / (b*(1+(1+(c*(p-a)/b))^(-1/c)))^2
	# fitlogdist <- fitdist(data=y, distr="logdist", start=list(a=10^(-69/10), b=10, c=3))


}
# fitdist_best(x,y)

############ LogLogistic ##################
fit_loglogistic = function(x,y) {
## INSTALL nplr
# sudo apt-get -y install r-base build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
# echo '
# install.packages("devtools")
# require(devtools)
# install_github("fredcommo/nplr")
# ' | sudo -i R --no-save --interactive 

	## RUN N-Par Logistic Regression
	require(nplr)
	#fitnpl <- nplr(x,y,useLog=TRUE)
	fitnpl <- nplr(x, y_mW, useLog=TRUE) #, LPweight=0)
	#getGoodness(fitnpl)
	#getStdErr(fitnpl)
	fitnpl
	plot(fitnpl, main=TITLE) #showSDerr=TRUE, lwd=4, cex.main=1.25")
	getPar(fitnpl)

# > fitnpl <- nplr(x, y_mW, useLog=TRUE)
# Testing pars...
# The 3-parameters model showed better performance
# > fitnpl
# Instance of class nplr 

# Call:
# nplr(x = x, y = y_mW, useLog = TRUE)
# weights method: residuals

# 3-P logistic model
# Bottom asymptote: 0 
# Top asymptote: 5.905423e-06 
# Inflexion point at (x, y): 1 2.952712e-06 
# Goodness of fit: 0.07901603 
# Weighted Goodness of fit: 0.999976 
# Standard error: 1.222196e-05 6.182529e-07 

# > getPar(fitnpl)
# $npar
# [1] 3

# $params
#   bottom          top xmid      scal s
# 1      0 5.905423e-06    1 -7.553113 1

}
# fit_loglogistic(x,y)


fit_3parLogNormal = function(x,y) {
## INSTALL bbmle
# echo 'install.packages("bbmle")' | sudo -i R --no-save --interactive 
	library(bbmle)

	mle2(y~dlnorm3(m,s,t), data=data.frame(-y_dB),start=list(m= 3, s = 10, t = -69), method="Nelder-Mead")
# Coefficients:
#            m            s            t 
#   0.09152264   4.95512825 -56.07357569 
# Log-likelihood: -398.26 
	summary(-(rlnorm3(n=1000, shape = 0.09152264, scale = 4.95512825, thres = -56.07357569))) 


	fitdist(data=-y_dB, dlnorm3, start=list(shape = 0.09152264, scale = 4.95512825, thres = -56.07357569))
# Fitting of the distribution ' lnorm3 ' by maximum likelihood 
# Parameters:
#           estimate  Std. Error
# shape   0.09121894  0.02849053
# scale   4.95529405  0.28055540
# thres -56.21239176 39.58281823

	model_lnorm3 <- fitdist(data=-y_dB, dlnorm3, start=list(shape = 0.09152264, scale = 4.95512825, thres = -56.07357569))
# Fitting of the distribution ' lnorm3 ' by maximum likelihood 
# Parameters : 
#           estimate  Std. Error
# shape   0.09121894  0.02849053
# scale   4.95529405  0.28055540
# thres -56.21239176 39.58281823

	summary(-(rlnorm3(n=1000, scale = 4.95512825, shape = 0.09152264, thres = -56.07357569))) 
# Loglikelihood:  -398.2582   AIC:  802.5164   BIC:  810.3319 
# Correlation matrix:
#            shape      scale      thres
# shape  1.0000000 -0.9737957  0.9742179
# scale -0.9737957  1.0000000 -0.9994712
# thres  0.9742179 -0.9994712  1.0000000
	
}
# fit_3parLogNormal(x,-y_dB)

fit_3parLogLogistic = function(x,y) {
## INSTALL FAdist
# echo 'install.packages("FAdist")' | sudo -i R --no-save --interactive 
	library(FAdist)
	library(MASS)


	mle2(y~dllog3(m,s,t), data=data.frame(y_mW),start=list(m= 0.03, s = 6, t = min(y_dB)), method="Nelder-Mead")
# Coefficients:
#            m            s            t 
#   0.03483763   5.23146324 -99.06433554 
# Log-likelihood: -389.26 
	Y <- -rllog3(n=1000, 0.03483763, 5.23146324, -99.06433554 )
	summary( Y )

	model_llog3 = fitdist(data=y_dB, dllog3, start=list(0.03483763, 5.23146324, -99.06433554))
# Fitting of the distribution ' llog3 ' by maximum likelihood 
# Parameters:
#      estimate Std. Error
# 1   0.6238972 0.08765614
# 2   2.1584165 0.13734617
# 3 -99.4197743 0.51712110

	## MANUAL TESTS
	# YmW_log3  <- rllog3(n=1000, shape = 2.37, scale = 15.30, thres = 10^(-100/10))
	# YdB_log3 <- -10*log10(YmW_log3)
	# summary(YdB_log3)
	# YdB_logis <- rlogis(n=1000, location = -67.817867, scale = 5.681117)
	# YmW_logis <- 10^(YdB_logis/10)
	# summary(YdB_logis)
	# YdB_gamma <- -rgamma(n=1000, shape=42.1, rate=0.62)
	# YmW_gamma
	# summary(Y_gamma)
}


fitinfo = function(fitmodel){
	summary(fitmodel) # show results
	
	#gofstat(fitmodel)
	
	# # Other useful functions 
	coefficients(fitmodel) # model coefficients
	confint(fitmodel, level=0.95) # CIs for model parameters 
	# #fitted(fitmodel) # predicted values
	# #residuals(fitmodel) # residuals
	# #anova(fitmodel) # anova table 
	# #vcov(fitmodel) # covariance matrix for model parameters 
	# #influence(fitmodel) # regression diagnostics

	# diagnostic plots 
	layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
	plot(fitmodel)
}

fit_all = function(x,y){
	model <- NULL
	fit <- NULL
	gof <- NULL

	dists <- c("norm", "lnorm", "pois", "exp", "gamma", "nbinom", "geom", "beta", "unif", "logis")
	metrics <- c("aic","bic","ks","cvm","ad")

	aic <- NULL
	bic <- NULL
	ks  <- NULL
	cvm <- NULL
	ad  <- NULL
	chisq  <- NULL


	# x <- wifi$dist
	# y <- y_dB
	# if( min(y) < 0 ){
	# 	## put y in dB logscale
	# 	#y = 10^(y/10)	
	# 	## put y in positive scale
	# 	y <- -y
	# }

	# model_norm  = fitdist(data=-y_dB, distr="norm")
	# model_logis = fitdist(data=-y_dB, distr="logis")
	for (dist in c("norm","logis")) {
		#print(dist)
		model <- fitdist(data=-y_dB, distr=dist)
		fit   <- gofstat(model)
		aic[dist] <- fit$aic
		bic[dist] <- fit$bic
		ks[dist]  <- fit$ks
		cvm[dist] <- fit$cvm
		ad[dist]  <- fit$ad
		chisq[dist]  <- fit$chisq
	}

	# model_lnorm = fitdist(data=-y_dB, distr="lnorm")
	# model_exp   = fitdist(data=-y_dB, distr="exp")
	# model_gamma = fitdist(data=-y_dB, distr="gamma")
	# model_unif  = fitdist(data=-y_dB, distr="unif")
	for (dist in c("lnorm","exp","gamma")) {
		#print(dist)
		model <- fitdist(data=-y_dB, distr=dist)
		fit   <- gofstat(model)
		aic[dist] <- fit$aic
		bic[dist] <- fit$bic
		ks[dist]  <- fit$ks
		cvm[dist] <- fit$cvm
		ad[dist]  <- fit$ad
		chisq[dist]  <- fit$chisq
	}

	aic <- sort(aic)
	bic <- sort(bic)
	ks  <- sort(ks)
	cvm <- sort(cvm)
	ad  <- sort(ad)
	chisq  <- sort(chisq)


	print(names(aic))
	print(names(bic))
	print(names(ks))
	print(names(cvm))
	print(names(ad))

	bestdist <- print(names(aic)[1])
	bestmodel <- fitdist(data=-y_dB, distr=bestdist)
	print(bestdist)

	# bestdist <- print(names(ad)[1])
	# bestmodel <- fitdist(data=-y_dB, distr=bestdist)
	# print(bestdist)

	summary(bestmodel) # show results

	gofstat(bestmodel)
	coefficients(bestmodel) # model coefficients
	confint(bestmodel, level=0.95) # CIs for model parameters 

	# diagnostic plots 
	layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
	plot(bestmodel)

	print(bestdist)

# > print(names(aic))
# [1] "logis" "norm"  "gamma" "lnorm" "exp"  
# > print(names(bic))
# [1] "logis" "norm"  "gamma" "lnorm" "exp"  
# > print(names(ks))
# [1] "norm"  "logis" "gamma" "lnorm" "exp"  
# > print(names(cvm))
# [1] "logis" "norm"  "gamma" "lnorm" "exp"  
# > print(names(ad))
# [1] "logis" "norm"  "gamma" "lnorm" "exp"  
# > aic
#     logis      norm     gamma     lnorm       exp 
#  775.5513  785.2824  803.6166  814.3864 1093.7767 
# > bic
#     logis      norm     gamma     lnorm       exp 
#  780.7616  790.4928  808.8270  819.5967 1096.3819 
# > bestdist <- print(names(aic)[1])
# [1] "logis"
# > bestmodel <- fitdist(data=-y_dB, distr=bestdist)
# > print(bestdist)
# [1] "logis"
# > summary(bestmodel) # show results
# Fitting of the distribution ' logis ' by maximum likelihood 
# Parameters : 
#           estimate Std. Error
# location 88.185128   1.082976
# scale     6.282901   0.532610
# Loglikelihood:  -385.7757   AIC:  775.5513   BIC:  780.7616 
# Correlation matrix:
#            location      scale
# location  1.0000000 -0.1173162
# scale    -0.1173162  1.0000000

# > gofstat(bestmodel)
# Goodness-of-fit statistics
#                              1-mle-logis
# Kolmogorov-Smirnov statistic   0.1517015
# Cramer-von Mises statistic     0.3470420
# Anderson-Darling statistic     2.9566927

# Goodness-of-fit criteria
#                                1-mle-logis
# Akaike's Information Criterion    775.5513
# Bayesian Information Criterion    780.7616
# > coefficients(bestmodel) # model coefficients
#  location     scale 
# 88.185128  6.282901 
# > confint(bestmodel, level=0.95) # CIs for model parameters 
#              2.5 %    97.5 %
# location 86.062534 90.307722
# scale     5.239005  7.326798
}