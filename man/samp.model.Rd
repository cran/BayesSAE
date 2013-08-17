\name{samp.model}
\alias{samp.model}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Sampling Model Class
%%  ~~function to do ... ~~
}
\description{This function aims to specify small area sampling model. 
Results are obtained as a new class.  
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
samp.model(y, vardir, innov = "normal", df = NULL, data)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{vector containing direct estimators for each domain. 
%%     ~~Describe \code{y} here~~
}
  \item{vardir}{vector containing sampling variances of direct estimators for each domain. The number of 
  domains must be the same as that of \code{y}. Values must be sorted as the variable \code{y}.
%%     ~~Describe \code{vardir} here~~
}
  \item{innov}{distribution of sampling innovation, to be chosen between \code{"normal"} or \code{"t"}. 
  Details are given under Details.
%%     ~~Describe \code{innov} here~~
}
  \item{df}{vector containing degrees of freedom of direct estimators for each domain if \code{"t"} distribution
  is adopted. This argument could be omitted if innovations are normal distributed. 
%%     ~~Describe \code{df} here~~
}
  \item{data}{an optional data frame containing variables \code{y} and \code{vardir}. Variables are taken from the 
  environment from which \code{samp.model} is called by default.
%%     ~~Describe \code{data} here~~
}
}
\details{Central Limit Theory indicates direct estimators are approximately normal distributed. Such models
correspond to the classical Fay-Herriot models. However, You and Chapman (2006) showed that sampling variance
would be underestimated by direct estimators resulting from small sample sizes within each domain. To 
overcome this problem, they made the sampling innovations \code{t} distributed and the You-Chapman model was thus 
constructed. 
%%  ~~ If necessary, more details than the description above ~~
}
\value{The function returns a object of class with the following attributes:

\item{y}{vector of direct estimators}
\item{vardir}{vector of variances of \code{y}}
\item{innov}{distribution of sampling innovations}
\item{df}{vector of d.f. of direct estimators if sampling innovations are \code{t} distributed}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
You, Y. and Chapman, B. (2006) Small Area Estimation Using Area Level Models and Estimated Sampling 
Variances. \emph{ Survey Methodology,} \bold{ 32:} 97-103.
}
\author{
%%  ~~who you are~~
Chengchun Shi
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{link.model}}, \code{\link{BayesSAE}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
#load data set
data(SAIPE)

#construct the sampling model
samp <- samp.model(y = SACPR, vardir = Vardir, data = SAIPE, innov = "t", 
   	df = rep(1, length(SAIPE$SACPR)))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ sampling model }
