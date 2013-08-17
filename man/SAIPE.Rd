\name{SAIPE}
\alias{SAIPE}
\docType{data}
\title{Data on U.S. Census Bureau's Small Area Income and Poverty Estimates (SAIPE) program 
%%   ~~ data name/kind ... ~~
}
\description{The U.S. Census Bureau's Small Area Income and Poverty Estimates (SAIPE) program
provides annual estimates of income and poverty statistics for all school districts, counties, and
states. The main objective of this program is to provide estimates of income and poverty for the
administration of federal programs and the allocation of federal funds to local jurisdictions.
The Data is a data frame containing 2005 ACS poverty rate estimators as well as its variance estimators,
and the common regressors used in SAIPE publications and conference papers William \emph{et. al} (2007) 
and Hawalay and Lahiriz (2012). 
%%  ~~ A concise (1-5 lines) description of the dataset. ~~
}
\usage{data(SAIPE)}
\format{
  A data frame with 51 observations on the following 7 variables.
  \describe{
    \item{\code{SACPR}}{2005 single year state-level poverty rate estimated by ACS(\%)}
    \item{\code{StdE}}{standard errors of ACSPR}
    \item{\code{SNAPR}}{state rate of Food Stamp participation}
    \item{\code{CenPR}}{U.S. census 2000 state-lvel poverty rate}
    \item{\code{CPER}}{state rate of IRS Child Tax-Poor Exemptions}
    \item{\code{CFR}}{IRS child filing state rate}
    \item{\code{Vardir}}{variance estimators of ACSPR}
  }
}
\source{\url{www.census.gov}
%%  ~~ reference to a publication or URL from which the data were obtained ~~
}
\references{
Bell, W. and Basel, W. and Cruse, C. and Dalzell, L. and Maples, J and O'Hara, B and Powers, D. (2007) 
\emph{ Use of ACS Data to Produce SAIPE Model-Based Estimates of Poverty for Counties,} U.S. Census official 
paper

Hawalay, S. and Lahiriz, P. (2012). \emph{ Hierarchical Bayes Estimation of Poverty Rates,} U.S. Census 
companion paper
%%  ~~ possibly secondary sources and usages ~~
}


