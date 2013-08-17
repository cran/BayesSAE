BayesSAE <-
function(samp, link, MCMCstart, prior, n){
     if (is.null(MCMCstart$theta) || is.null(MCMCstart$beta))
          stop("initial values for theta's and beta's must be provieded")
     if (link$spatial){ 
          if (is.null(MCMCstart$lambda)||(MCMCstart$lambda <= 0)||(MCMCstart$lambda >= 1))
               stop("initial value for lambda must be provided and bounded by 0 and 1")
     }
     if (is.null(MCMCstart$beta)||is.null(MCMCstart$theta))
          stop("Initial values for beta's and theta's should be provided")
     y <- samp$y
     phi <- samp$vardir
     innov <- samp$innov
     X <- link$X
     spatial <- link$spatial
     tran <- link$tran
     b <- link$b
     m <- length(y)
     p <- ncol(X)
     beta <- MCMCstart$beta
     theta <- MCMCstart$theta
     betatype <- prior$betatype
     betaprior <- prior$betaprior
     Sqsigmavtype <- prior$Sqsigmavtype
     Sqsigmavprior <- prior$Sqsigmavprior
     if (betatype != "normal" && betatype != "non_in")
          stop("only normal or non-informative distribution could be prior for beta's at present")
     if (Sqsigmavtype != "inv_gamma" && Sqsigmavtype != "unif")
          stop("only invert gamma or uniform distribution could be prior for Sqsigmav at present")
     if (betatype == "normal")
          betatype <- 0
     else
          betatype <- 1
     if (Sqsigmavtype == "inv_gamma")
          Sqsigmavtype <- 0
     else
          Sqsigmavtype <- 1
     if (nrow(X) != m)
          stop("number of rows of X and length of y are inconsistent")
     if (betatype == 0){
          beta0 <- betaprior$beta0
          if (length(beta0) != p)
               stop("number of rows of X and length of beta0 are inconsistent")
          eps1 <- betaprior$eps1
          if (eps1 <= 0)
               stop("eps1 must be positive")
          betaprior <- c(beta0, eps1)
     }
     else
          betaprior <- 0
     if (Sqsigmavtype == 0){
          a0 <- Sqsigmavprior$a0
          b0 <- Sqsigmavprior$b0
          if (a0 <= 0 || b0 <= 0)
               stop("a0 and b0 must be positive")
          Sqsigmavprior <- c(a0 ,b0)
     }
     else{
          Sqsigmavprior <- Sqsigmavprior$eps2
          if(Sqsigmavprior <= 0)
               stop("eps2 must be positive")
     }
     if (spatial){
          prox <- link$prox
          li1 <- prox[,1] 
          li2 <- prox[,2] 
          num <- rep(0, m)
          for (i in 1:m){
               num[i] = sum(li1 == i) + sum(li2 == i)
          }
          lambda <- MCMCstart$lambda
          if (innov == "normal")
               result <- BayesSFH(theta, beta, lambda, y, t(X), phi, li1, li2, num, n, betaprior, Sqsigmavprior, betatype, Sqsigmavtype)
          else{ 
               ai <- prior$Sqsigmaprior$ai
               bi <- prior$Sqsigmaprior$bi
               if (any(ai <= 0) || any(bi <= 0))
                    stop("elements in ai and bi must be positive")
               if (length(ai) != m || length(bi) != m)
                    stop("lengths of ai and bi should be the same as that of y")
               df <- samp$df
               result <- BayesSYC(theta, beta, lambda, y, t(X), phi, li1, li2, num, n, betaprior, Sqsigmavprior, c(ai, bi, df), betatype, Sqsigmavtype)
          }
     }
     else{
          if (tran == "log" && any(theta <= 0))
               stop("initial values for theta's should be positive")
          if (tran == "logit" && any(theta <= 0 || theta >= 1))
               stop("initial values for theta's should be bounded by 0 and 1") 
          if (innov == "normal"){
               if (tran == "F")
                    result <- BayesFH(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, betatype, Sqsigmavtype)
               else if (tran == "log")
                    result <- BayesUFH(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, betatype, 1, Sqsigmavtype)
               else
                    result <- BayesUFH(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, betatype, 2, Sqsigmavtype)
          }
          else{
               ai <- prior$Sqsigmaprior$ai
               bi <- prior$Sqsigmaprior$bi
               if (any(ai <= 0) || any(bi <= 0))
                    stop("elements in ai and bi must be positive")
               if (length(ai) != m || length(bi) != m)
                    stop("lengths of ai and bi should be the same as that of y")
               df <- samp$df
               if (tran == "F")
                    result <- BayesYC(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, c(ai, bi, df), betatype, Sqsigmavtype)
               else if (tran == "log")
                    result <- BayesUYC(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, c(ai, bi, df), betatype, 1, Sqsigmavtype)
               else
                    result <- BayesUYC(theta, beta, y, t(X), b, phi, n, betaprior, Sqsigmavprior, c(ai, bi, df), betatype, 2, Sqsigmavtype)
          }
     }
     result
}
