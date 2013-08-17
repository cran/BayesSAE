model.compare <-
function(result, samp, subset, poest = "mean"){
     if (samp$innov != "normal" && samp$innov != "t")
          stop("Sampling model must be with normal or t innovations") 
     if (is.null(result$theta))
          stop("MCMC draws of theta's are needed")
     if (poest != "mean" && poest != "median")
          stop("only posterior means or medians can be used as the point estimator for theta's at present")
     theta <- result$theta[,subset]
     m <- nrow(theta)
     n <- ncol(theta)
     y <- samp$y
     if (length(y) != m)
          stop("length of y and number of rows of theta should be the same")
     if (samp$innov == "normal"){
          vardir <- samp$vardir
          if (length(vardir) != m)
               stop("length of vardir and number of rows of theta should be the same")
          D_avg <- -2 * mean(apply(dnorm(y, theta, sqrt(vardir), log = TRUE), 2, sum))
          if (poest == "mean")
               D_theta.hat <- dnorm(y, rowMeans(theta), sqrt(vardir), log = TRUE)
          else
               D_theta.hat <- dnorm(y, apply(theta, 1, median), sqrt(vardir), log = TRUE) 
          D_theta.hat <- -2 * sum(D_theta.hat)  
     }
     else{
          Sqsigma <- result$Sqsigma[,subset]
          if (nrow(Sqsigma) != m)
               stop("number of rows of Sqsigma should be the same as that of theta")
          if (ncol(Sqsigma) != n)
               stop("number of draws of Sqsigma should be the same as that of theta")
          D_avg <- -2 * mean(apply(dnorm(y, theta, sqrt(Sqsigma), log = TRUE), 2, sum))
          if (poest == "mean")
               D_theta.hat <- dnorm(y, rowMeans(theta), sqrt(rowMeans(Sqsigma)), log = TRUE)
          else
               D_theta.hat <- dnorm(y, apply(theta, 1, median), sqrt(apply(Sqsigma, 1, median)), log = TRUE) 
          D_theta.hat <- -2 * sum(D_theta.hat) 
     }
     criter <- list(D_avg = D_avg, D_theta.hat = D_theta.hat, DIC = 2 * D_avg - D_theta.hat)
}
