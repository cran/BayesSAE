\name{BayesSAE}
\alias{BayesSAE}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{MCMC Draws Based on Area-Level Models 
%%  ~~function to do ... ~~
}
\description{This function obtains MCMC draws of specific small area area-level models defined by the 
\code{samp.model} and \code{link.model} function. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
BayesSAE(samp, link, MCMCstart, prior, n)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{samp}{a list returned by the function \code{\link{samp.model}}. 
%%     ~~Describe \code{samp} here~~
}
  \item{link}{a list returned by the function \code{\link{link.model}}. 
%%     ~~Describe \code{link} here~~
}
  \item{MCMCstart}{a list specifying MCMC initial values of parameters. See Details.
%%     ~~Describe \code{MCMCstart} here~~
}
  \item{prior}{a list of objects specifying priors. See Details.
%%     ~~Describe \code{prior} here~~
}
  \item{n}{user-specified number of MCMC draws. 
%%     ~~Describe \code{n} here~~
}
}
\details{The argument \code{MCMCstart} should contain the following components:

\itemize{
\item \code{theta}: vector of initial values of the variable of interest for each domain. The number of domains
should be the same as that of direct estimators. And values must be sorted as direct estimators.
\item \code{beta}: vector of initial values of the regression coefficient. 
}

And another attribute \code{lambda} bounded by 0 and 1 can be determined as the initial value in the spatial model. 
See more details about model parameters in You and Chapman (2006). 

Initial values are crucial to MCMC convergence. EBLUP predictors of \eqn{\theta}'s and \eqn{\beta}'s provides good 
starting values for MCMC procedure. 

The list \code{prior} should include following attributes for basic Fay-Herriot model:

\itemize{
\item \code{betatype}: to be chosen between \code{"non_in"} or \code{"normal"}. If \code{betatype = "non_in"}, 
non-informative prior would be specified for \eqn{\beta}. Otherwise, prior for \eqn{\beta} would be normal 
distribution. \item \code{betaprior}: a list contains components \code{beta0} and \code{eps1} if \code{betatype = "normal"}. 
As a result \eqn{\beta} will be distributed with mean \eqn{\mu = \beta_0} and covariance matrix 
\eqn{\Sigma = diag(rep(1/eps1, p))} where p is length of \eqn{\beta} including the intercept term. 
\item \code{Sqsigmavtype}: to be chosen between \code{"inv_gamma"} and \code{"unif"}. \eqn{\sigma_v^2} is the 
variance of residual in the linking model. See \code{\link{link.model}}. If \code{Sqsigmavtype = "inv_gamma"}, 
inverse gamma prior would be specified for \eqn{\sigma_v^2}. Otherwise, \eqn{\sigma_v^2} would be uniformly distributed.
\item \code{Sqsigmavprior}: a list containing components \code{a0} and \code{b0} as shape and rate parameter in the 
gamma prior. if \code{Sqsigmavtype = "inv_gamma"}. Otherwise the list should contain the \code{eps2} component and 
\eqn{\sigma_v^2} would be uniformly distributed on (0, 1 / \code{eps2})
}

Besides, the \code{prior} list should also include attribute \code{Sqsigmaprior} to specify priors for \eqn{\sigma_i^2} 
in the You-Chapman model. See You and Chapman (2006). The \code{Sqsigmaprior} is also a list contains the 
the components \code{ai} and \code{bi}. Both \code{ai} and \code{bi} are vectors whose length are the same as 
number of domains. Thus, prior for \eqn{\sigma_i^2} would be inverse gamma distribution with shape parameter 
\code{ai[i]} and rate parameter \code{bi[i]}. 
%%  ~~ If necessary, more details than the description above ~~
}
\value{The function returns a list containing the following components:

\item{theta}{\code{m * n} matrix of MCMC draws for \eqn{\theta}'s} 
\item{beta}{\code{p * n} matrix of MCMC draws for \eqn{\beta}'s where \code{p} is the number of regressors 
including the intercept term}
\item{Sqsigmav}{vector contains \code{n} draws for \eqn{\sigma_v^2}}
\item{type}{character string indicating the type of the model. For instance "UFH" indicates that it's an
unmatched Fay-Herriot model while "SYC" refers to spatial You-Chapman model}
\item{lambda}{vector contains \code{n} draws for \eqn{\lambda} in the spatial model}
\item{rate_lam}{the acceptance rate of \eqn{\lambda} since draws of \eqn{\lambda} are generated by M-H algorithm} 
\item{Sqsigma}{\code{m * n} matrix of MCMC draws for \eqn{\sigma_i^2} in the You-Chapman model}

If it's an unmatched model, \eqn{\theta_i}'s are generated by M-H algorithm and \code{rate} as a 
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
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
Chengchun Shi
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{samp.model}}, \code{\link{link.model}}
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{Hierarchical Bayes}