\name{y.rep}
\alias{y.rep}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Replicated Data
%%  ~~function to do ... ~~
}
\description{This function draws replicated data from posterior predictive distributions. With these 
replications, one can display graphical posterior checks or compute the Bayesian p-value to see whether
the model fits the data well. See details in Gelman \emph{et. al} (2006). 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
y.rep(result, samp, subset, repperdr = 1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{result}{MCMC draws obtained by \code{\link{BayesSAE}} function. 
%%     ~~Describe \code{result} here~~
}
  \item{samp}{list returned by the function \code{samp.model}. 
%%     ~~Describe \code{samp} here~~
}
  \item{subset}{vector specifying subset of MCMC draws used to obtain Rao-Blackwell estimators. 
  If omitted, all draws are utilized. See also \code{\link{theta.HB}}
%%     ~~Describe \code{subset} here~~
}
  \item{repperdr}{number of replicated data each posterior draw generated. See also Details.
%%     ~~Describe \code{repperdr} here~~
}
}
\details{
Let \eqn{\theta} denote all the parameters in the model, and \eqn{\theta^{i}, i = 0, 1, \ldots, n} be the 
\code{n} posterior draws. Supposing the argument repperdr is 5, and then 5 replications would be obtained 
from the distribution \eqn{p(y|\theta^{i})} for each \eqn{i}. 
%%  ~~ If necessary, more details than the description above ~~
}
\value{
\code{m * k} matrix of replicated data where \code{m} is the number of domains. Values are sorted as the 
direct estimators. 
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
\code{\link{BayesSAE}}, \code{\link{theta.HB}}
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

#obtain replicated data
y_rep <- y.rep(result, samp, 3001:5000, 1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
