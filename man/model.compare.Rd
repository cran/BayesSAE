\name{model.compare}
\alias{model.compare}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Criteria for Bayesian Model Comparison
%%  ~~function to do ... ~~
}
\description{This function provides criteria such as \eqn{D_{avg}}, \eqn{D_{\hat{\theta}}}{D_{\theta}} and 
\code{DIC} for specified area-level models. Details could be found in Gelman \emph{et. al} (2006).  
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
model.compare(result, samp, subset, poest = "mean")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{result}{MCMC draws obtained by \code{BayesSAE} function. See \code{\link{BayesSAE}}
%%     ~~Describe \code{result} here~~
}
  \item{samp}{list generate by \code{samp.model} function. See \code{\link{samp.model}} 
%%     ~~Describe \code{samp} here~~
}
  \item{subset}{vector specifying subset of MCMC draws used to get criteria. If omitted, all draws are
  utilized. See also \code{\link{theta.HB}}
%%     ~~Describe \code{subset} here~~
}
  \item{poest}{to be chosen between \code{"mean"} and \code{"median"}. If \code{poest = "mean"}, 
  point estimator of \eqn{\theta} will be the posterior mean of draws of \eqn{\theta}'s. Otherwise, it will 
  be the posterior median. 
%%     ~~Describe \code{poest} here~~
}
}

\value{
This function returns a list containing the following components:

\item{D_avg}{average deviance, defined as \deqn{-\frac{2}{n} \sum_{i=1}^n p(y|\theta^{i})}{%
-2/n \sum_{i=1}^n p(y|\theta^{i})} where \eqn{\theta^{i}} donates the ith posterior draw of \eqn{\theta}.}
\item{D_theta.hat}{discrepancy between data and model depending on a point estimator for \eqn{\theta},
defined as \deqn{-2 p(y|\hat{\theta})}{-2 p(y|\theta^')} where \eqn{\hat{\theta}}{\theta^'} is the point 
estimator for \eqn{\theta}}
\item{DIC}{deviance information criterion, defined as \code{2D_avg - D_theta.hat}. Small \code{DIC} value 
indicates strong out-of-sample predictive power}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
Gelman, A. and Carlin, J. B. and Stern, H. S. and Rubin, D. B. (2006). \emph{ Bayesian Data Analysis,} 
CRC Press Company.
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
Chengchun Shi
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{theta.HB}}, \code{\link{BayesSAE}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
#load data set
data(SAIPE)

#construct the sampling model
samp <- samp.model(y = SACPR, vardir = Vardir, data = SAIPE, innov = "t", 
   	df = rep(50, length(SAIPE$SACPR)))

#define the neighbourhood matrix
prox <- cbind(sample(1:51, 50, replace = TRUE), sample(1:51, 50, replace = TRUE))
prox <- prox[prox[,1] != prox[,2], ]

#construct the linking model
link <- link.model(~SNAPR+CenPR+CPER, data = SAIPE, prox = prox, spatial = TRUE)

#get initial values
MCMCstart <- list(theta = runif(51), beta = runif(4), lambda = runif(1))

#specify priors
Sqsigmavprior <- list(a0 = 0.1^6, b0 = 0.1^6)
Sqsigmaprior <- list(ai = rep(0.1^6, 51), bi = rep(0.1^6, 51))
prior <- list(betatype = "non_in", Sqsigmavtype = "inv_gamma", Sqsigmavprior = Sqsigmavprior, 
   	Sqsigmaprior = Sqsigmaprior)

#obtain MCMC draws
result <- BayesSAE(samp, link, MCMCstart, prior, 5000)	
	
#obtain criteria
cri <- model.compare(result, samp, 3001:5000)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
