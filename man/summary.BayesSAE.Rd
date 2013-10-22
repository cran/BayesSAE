\name{summary.BayesSAE}
\alias{summary.BayesSAE}
\alias{print.summary.BayesSAE}
\alias{MCMC}
\title{Extract Information from Object BayesSAE}
\description{Methods for extracting information from fitted simplex regression model
objects of class \code{"BayesSAE"}}
\usage{
\method{summary}{BayesSAE}(object, HB = TRUE, ...)

MCMC(object, ...)
}
\arguments{
	\item{object}{fitted model object of class "BayesSAE"}
	\item{HB}{logical variable indication whether Rao-Blackwellization of \code{theta} (\code{HB = TRUE})
	or posterior mean of \code{theta} (\code{HB = FALSE}) should be included. See Rao (2003) for details}	
	\item{...}{currently not used}
}

\value{
These functions aim to extract information from the object of class \code{"BayesSAE"}. Posterior means as 
and quantiles of regression coefficients as well as variance of residuals are included in the \code{summary}
output. Function \code{mcmc} provides the object of class \code{mcmc} in the coda package for posterior 
diagnosis. 
}

\references{
Rao, J. N. K. (2003) \emph{ Small Area Estimation}. John Wiley and Sons.
}

\author{
%%  ~~who you are~~
Chengchun Shi
}

\examples{
# load data set
data(SAIPE)

# obtain posterior draws
result <- BayesSAE(SACPR~SNAPR+CenPR+CPER|Vardir, data = SAIPE, mcmc = 5000)

# summary information
summary(result)
}