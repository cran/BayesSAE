y.rep <-
function(result, samp, subset, repperdr = 1){
     if (samp$innov != "normal" && samp$innov != "t")
          stop("Sampling model must be with normal or t innovations") 
     if (is.null(result$theta))
          stop("MCMC draws of theta's are needed")
     theta <- result$theta[,subset]
     m <- nrow(theta)
     n <- ncol(theta)
     if (repperdr != as.integer(repperdr) || repperdr <= 0)
          stop("replication per draw must be positive integers")
     theta <- array(apply(theta, 2, rep, repperdr), c(m, n * repperdr))
     eps <- array(rnorm(m * n * repperdr), c(m, n * repperdr))
     if (samp$innov == "normal"){
          vardir <- samp$vardir
          if (length(vardir) != m)
               stop("length of vardir and number of rows of theta should be the same")
          theta <- theta + vardir * eps  
     }
     else{
          Sqsigma <- result$Sqsigma[,subset]
          if (nrow(Sqsigma) != m)
               stop("number of rows of Sqsigma should be the same as that of theta")
          if (ncol(Sqsigma) != n)
               stop("number of draws of Sqsigma should be the same as that of theta")
          Sqsigma <- array(apply(Sqsigma, 2, rep, repperdr), c(m, n * repperdr))
          theta <- theta + Sqsigma * eps
     }
     theta
}
