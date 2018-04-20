\name{replication}
\alias{replication}
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
replication(object, repperdr = 1, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{an object of class obtained by \code{\link{BayesSAE}} function. 
%%     ~~Describe \code{result} here~~
}

  \item{repperdr}{number of replicated data each posterior draw generated. See also Details.
%%     ~~Describe \code{repperdr} here~~
}
  \item{...}{currently not used}
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
\code{\link{BayesSAE}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
#load data set
data(SAIPE)

#obtain MCMC draws
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000)

#obtain replicated data
replication(result)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
