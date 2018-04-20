\name{BayesSAE}
\alias{BayesSAE}
\alias{print.BayesSAE}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{MCMC Draws Based on Area-Level Models 
%%  ~~function to do ... ~~
}
\description{This function obtains MCMC draws of specific small area area-level models defined by the 
sampling model and linking model. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
BayesSAE(formula, innov = "normal", df = NULL, b = NA, spatial = FALSE, tran = 
     "F", prox = NULL, beta.start = NULL, theta.start = NULL, lam.start = runif(1), 
     prior = NULL, mcmc = 5000, burnin = 2500, thin = 5, data)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a symbolic description of the model to be fitted (of type y ~ x | z). \code{y} is the
  response variable in the sampling model while \code{x} is the design matrix in the linking model, 
  and \code{z} is the estimated variance of direct estimation in the sampling model. See Rao (2003) 
  for details about the sampling model and linking model in small area estimation.
} 
  \item{innov}{distribution of innovations in the sampling model. to be chosen between \code{"normal"}
  and "t". 
} 
  \item{df}{a vector containing degrees of freedom for the \code{t} innovation in the sampling model
  if \code{innov = "t"}. 
%%     ~~Describe \code{MCMCstart} here~~
}
  \item{b}{an optional weights vector to be used in the fitting process. Number of domains must be 
  the same as that of the direct estimators. By default is a vector of ones. See Details.
}  
  \item{spatial}{a logical variable indicating whether it's a spatial model or not.
}  
  \item{tran}{the transformation to be taken on the responsive variable in the linking model, to be chosen
  between \code{"F"}, \code{"log"} or \code{"logit"}. \code{"F"} is the default value and indicates no 
  transformation taken. See Details.
}
  \item{prox}{\code{l * 2} matrix defining the neighbourhood matrix. See also Details.
}
  \item{beta.start}{initial values for beta's. By default is the coefficients of regression model
  where the response vector is directly regressed on the design matrix in the linking model.
}
  \item{theta.start}{initial values for theta's. By default is the response vector.  
}
  \item{lam.start}{initial value for lambda in the spatial model. See You and Chapman (2006).
}
  \item{prior}{a list of objects specifying priors. See Details.
%%     ~~Describe \code{prior} here~~
}
  \item{mcmc}{user-specified number of MCMC draws. 
%%     ~~Describe \code{n} here~~
}
  \item{burnin}{The number of burnin iterations for the sampler. See Gelman (2006). 
}
  \item{thin}{the thinning interval used in the simulation. See Gelman (2006). 
}
  \item{data}{an optional data frame, list or environment containing variables in the model.
}
}
\details{
Let \eqn{\theta_i} donates variable of interest for each domain i, \eqn{x_i} the regressors, 
\eqn{\beta} the regression coefficient, \eqn{v_i} i.i.d normal innovations. If argument \code{b} is 
specified, the linking model is of the form: \eqn{\theta_i = x_i \beta + b_i v_i}. 

If \code{tran = "log"}, the linking model will be: \eqn{\log(\theta_i) = x_i \beta + b_i v_i}. 
\code{tran = "logit"} means that logit transformation will be taken and the model will be: \eqn{logit(\theta_i) 
= x_i \beta + b_i v_i}. Both are unmatched area level models. See Rao (2003).

The neighbourhood matrix has the ith diagonal element equal to the number of neighbours of area i, and 
off-diagonal elements equal to \code{-1} if the corresponding areas are neighbours otherwise \code{0}. 
See You and Chapman (2006). 

The ith tuple in the argument \code{prox} indicates that area \code{prox[i, 1]} and area \code{prox[i, 2]} 
are neighbours. Duplicated tuples will be omitted. For example, if the first row of \code{prox} is (1, 2) 
while the second is (2, 1), the second row will be deleted. The two elements within each tuple are not 
supposed to be the same. 

Initial values are crucial to MCMC convergence. EBLUP predictors of \eqn{\theta}'s and \eqn{\beta}'s can provides 
good starting values for MCMC procedure. 

The list \code{prior} should include following attributes for basic Fay-Herriot model:

\itemize{
\item \code{beta.type}: to be chosen between \code{"non_in"} or \code{"normal"}. If \code{beta.type = "non_in"}, 
non-informative prior would be specified for \eqn{\beta}. Otherwise, prior for \eqn{\beta} would be normal 
distribution. 
\item \code{beta.prior}: a list contains components \code{beta0} and \code{eps1} if \code{beta.type = "normal"}. 
As a result \eqn{\beta} will be distributed with mean \eqn{\mu = \beta_0} and covariance matrix 
\eqn{\Sigma = diag(rep(1/eps1, p))} where p is length of \eqn{\beta} including the intercept term. 
\item \code{sigv.type}: to be chosen between \code{"inv_gamma"} and \code{"unif"}. \eqn{\sigma_v^2} is the 
variance of residual in the linking model. If \code{sigv.type = "inv_gamma"}, inverse gamma prior would be specified 
for \eqn{\sigma_v^2}. Otherwise, \eqn{\sigma_v^2} would be uniformly distributed. 
\item \code{sigv.prior}: a list containing components \code{a0} and \code{b0} as shape and rate parameter in the 
gamma prior. if \code{sigv.type = "inv_gamma"}. Otherwise the list should contain the \code{eps2} component and 
consequently \eqn{\sigma_v^2} would be uniformly distributed on (0, 1 / \code{eps2})
}

Besides, the \code{prior} list should also include attribute \code{sig2.prior} to specify priors for \eqn{\sigma_i^2} 
in the You-Chapman model. See You and Chapman (2006). The \code{sig2.prior} is also a list contains the 
components \code{ai} and \code{bi}. Both \code{ai} and \code{bi} are vectors whose length are the same as 
number of domains. Thus, prior for \eqn{\sigma_i^2} would be inverse gamma distribution with shape parameter 
\code{ai[i]} and rate parameter \code{bi[i]}. Default value of elements in \code{ai} and \code{bi} are all 0.05. 
%%  ~~ If necessary, more details than the description above ~~
}
\value{The function returns a object of class \code{"BayesSAE"} containing the following components:

\item{mcmc}{an mcmc object that contains the posterior sample. This object can be summarized
by functions provided by the coda package} 
\item{type}{character string indicating the type of the model. For instance "UFH" indicates that it's an
unmatched Fay-Herriot model while "SYC" refers to spatial You-Chapman model}
\item{Y}{response vector in the sampling model}
\item{X}{design matrix in the linking model}
\item{Z}{variance of direct estimation in the sampling model}
\item{lam.rate}{the acceptance rate of \eqn{\lambda} since draws of \eqn{\lambda} are generated by M-H algorithm} 
\item{D_avg}{average deviance, defined as \deqn{-\frac{2}{n} \sum_{i=1}^n p(y|\theta^{i})}{%
-2/n \sum_{i=1}^n p(y|\theta^{i})} where \eqn{\theta^{i}} donates the ith posterior draw of \eqn{\theta}.}
\item{D_theta.hat}{discrepancy between data and model depending on a point estimator for \eqn{\theta},
defined as \deqn{-2 p(y|\hat{\theta})}{-2 p(y|\theta^')} where \eqn{\hat{\theta}}{\theta^'} is the point 
estimator for \eqn{\theta}. Here we calculate the posterior mean as the point estimator}
\item{DIC}{deviance information criterion, defined as \code{2D_avg - D_theta.hat}. Small \code{DIC} value 
indicates strong out-of-sample predictive power}
\item{theta.HB}{a vector of length the same as number of domains provides Rao-Blackwell estimators for each area.}
\item{call}{the original function call}
\item{HB}{Rao-Blackwellization of theta's}
\item{spatial}{logical variable indicating whether the model is spatial or not}
\item{tran}{character indicating the transformation of response variable in the linking model}

If it's an unmatched model, \eqn{\theta_i}'s are generated by M-H algorithm and \code{theta.rate} as a 
vector of length \code{m} provides acceptance rate for each \eqn{\theta_i} respectively is involved 

%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
You, Y. and Chapman, B. (2006) Small Area Estimation Using Area Level Models and Estimated Sampling 
Variances. \emph{ Survey Methodology,} \bold{ 32:} 97-103.

Rao, J. N. K. (2003) \emph{ Small Area Estimation}. John Wiley and Sons.

Gelman, A. and Carlin, J. B. and Stern, H. S. and Rubin, D. B. (2006). \emph{ Bayesian Data Analysis,} 
CRC Press Company.
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
Chengchun Shi
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\examples{
# load data set
data(SAIPE)
m <- length(SAIPE$SACPR)

# basic Fay-Herriort models (FH)
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000)

# You-Chapman models (YC)
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000, innov = "t", 
    df = rep(50, m))
	
# spatial model with unknown sampling variance (SYC)
# define the neighbourhood matrix
prox <- cbind(sample(1:51, 50, replace = TRUE), sample(1:51, 50, replace = TRUE))
prox <- prox[prox[,1] != prox[,2], ]
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000, innov = "t",
    df = rep(50, m), spatial = TRUE, prox = prox)

# Unmatched models (UFH)
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000, tran = "log")	
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Hierarchical Bayes}
