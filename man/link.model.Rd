\name{link.model}
\alias{link.model}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Linking Model Class
%%  ~~function to do ... ~~
}
\description{This function aims to specify small area linking model. 
Results are obtained as a new class. 
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
link.model(formula, weights, spatial = FALSE, tran = "F", data, prox = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{an object of class \code{\link{formula}}: a symbolic description of the model to be fitted. 
  Formula of the form \code{~ terms} where \code{terms} is a series of terms which specifies a linear 
  predictor for the variable of interest within each domain is enough to complete the model.  
%%     ~~Describe \code{formula} here~~
}
  \item{weights}{an optional weights vector to be used in the fitting process. Number of domains must be 
  the same as that of the direct estimators. By default is a vector of ones. See Details.
%%     ~~Describe \code{weights} here~~
}
  \item{spatial}{a logical variable indicating whether it's a spatial model or not.
%%     ~~Describe \code{spatial} here~~
}
  \item{tran}{the transformation to be taken on the responsive variable in the linking model, to be chosen
  between \code{"F"}, \code{"log"} or \code{"logit"}. \code{"F"} is the default value and indicates no 
  transformation taken. See Details.
%%     ~~Describe \code{tran} here~~
}
  \item{data}{optional data frame containing variable in the formula. Variables are taken from the 
  environment from which the function is called by default.
%%     ~~Describe \code{data} here~~
}
  \item{prox}{\code{l * 2} matrix defining the neighbourhood matrix. See also Details.
%%     ~~Describe \code{prox} here~~
}
}
\details{Let \eqn{\theta_i} donates variable of interest for each domain i, \eqn{x_i} the regressors, 
\eqn{\beta} the regression coefficient, \eqn{v_i} i.i.d normal innovations. If argument \code{weights} is 
specified, the linking model is of the form: \eqn{\theta_i = x_i \beta + weights_i v_i}. 

If \code{tran = "log"}, the linking model will be: \eqn{\log(\theta_i) = x_i \beta + weights_i v_i}. 
\code{tran = "logit"} means that logit transformation will be taken and the model will be: \eqn{logit(\theta_i) 
= x_i \beta + weigths_i v_i}. Both are unmatched area level models. See Rao (2003).

The neighbourhood matrix has the ith diagonal element equal to the number of neighbours of area i, and 
off-diagonal elements equal to \code{-1} if the corresponding areas are neighbours otherwise \code{0}. 
See You and Chapman (2006). 

The ith tuple in the argument \code{prox} indicates that area \code{prox[i, 1]} and area \code{prox[i, 2]} 
are neighbours. Duplicated tuples will be omitted. For example, if the first row of \code{prox} is (1, 2) 
while the second is (2, 1), the second row will be deleted. The two elements within each tuple are not 
supposed to be the same. 
%%  ~~ If necessary, more details than the description above ~~
}
\value{The function returns a object of class with the following attributes

\item{X}{matrix of regressors}
\item{b}{vector of the weights to be used in the fitting process}
\item{spatial}{logical variable indicating whether the model is spatial or not}
\item{tran}{character variable indicating the transformation to be taken on the responsive variable}
\item{prox}{\code{l * 2} matrix defining the neighbourhood matrix}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
Rao, J. N. K. (2003) \emph{ Small Area Estimation}. John Wiley and Sons.

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
\code{\link{samp.model}}, \code{\link{BayesSAE}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
#load date set
data(SAIPE) 

#define the neighbourhood matrix
prox <- cbind(sample(1:51, 50, replace = TRUE), sample(1:51, 50, replace = TRUE))
prox <- prox[prox[,1] != prox[,2], ]

#construct the linking model
link <- link.model(~SNAPR+CenPR+CPER, data = SAIPE, prox = prox, spatial = TRUE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{linking model}
