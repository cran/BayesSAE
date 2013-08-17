theta.HB <-
function(result, samp, link, subset){
     if (result$type!="FH"&&result$type!="YC"&&result$type!="SFH"&&result$type!="SYC")
          stop("such model couldn't get rao-blackwell estimators at present")
     if (is.null(result$beta)||is.null(result$Sqsigmav))
          stop("MCMC draws of beta's and Sqsigmav are needed")
     if (length(result$Sqsigmav) != ncol(result$beta))
          stop("number of draws of Sqsigmav and beta should be the same")
     beta <- result$beta[,subset]
     Sqsigmav <- result$Sqsigmav[subset]
     y <- samp$y
     X <- link$X
     m <- nrow(X)
     if (length(y) != m)
          stop("length of y and number of rows of X should be the same")
     if (result$type == "FH"){
          vardir <- samp$vardir
          b <- link$b
          if (length(b) != m)
               stop("length of b and number of rows of X should be the same")
          phi <- outer(1/b, Sqsigmav, "/")
          phi <- phi + 1 / vardir
          phi <- 1 / (vardir * phi)
          theta_HB <- phi * y + (1 - phi) * (X %*% beta)
          theta_HB <- rowMeans(theta_HB)
     }
     else if (result$type == "YC"){
          Sqsigma <- result$Sqsigma[,subset]
          b <- link$b
          if (length(b) != m)
               stop("length of b and number of rows of X should be the same")
          if (nrow(Sqsigma) != m)
               stop("number of rows of Sqsigma should be the same as that of beta")
          if (ncol(Sqsigma) != length(Sqsigmav))
               stop("number of draws of Sqsigma should be the same as that of Sqsigmav")
          phi <- outer(1/b, Sqsigmav, "/")
          phi <- phi + 1 / Sqsigma
          phi <- 1 / (Sqsigma * phi)
          theta_HB <- phi * y + (1 - phi) * (X %*% beta)
          theta_HB <- rowMeans(theta_HB) 
     }
     else if (result$type == "SFH"){
          vardir <- samp$vardir
          lambda <- result$lambda[subset]
          if (length(lambda) != length(Sqsigmav))
               stop("number of draws of lambda should be the same as that of Sqsigmav")
          prox <- link$prox
          R <- array(0, c(m, m))
          R[prox] <- -1
          li1 <- prox[,1] 
          li2 <- prox[,2]
   	   	  R[cbind(li2, li1)] <- -1
          num <- rep(0, m)
          for (i in 1:m){
               num[i] = sum(li1 == i) + sum(li2 == i)
          }
          diag(R) <- num
          theta_HB = rowMeans(apply(rbind(beta, Sqsigmav, lambda), MARGIN = 2, FUN = theta.HBSFH, y = y, X1 = X, R = R, vardir = vardir))
     }
     else{
          Sqsigma <- result$Sqsigma[,subset]
          lambda <- result$lambda[subset]
          if (nrow(Sqsigma) != m)
               stop("number of rows of Sqsigma should be the same as that of beta")
          if (ncol(Sqsigma) != length(Sqsigmav))
               stop("number of draws of Sqsigma should be the same as that of Sqsigmav")
          if (length(lambda) != length(Sqsigmav))
               stop("number of draws of lambda should be the same as that of Sqsigmav")
          prox <- link$prox
          R <- array(0, c(m, m))
          R[prox] <- -1
          li1 <- prox[,1] 
          li2 <- prox[,2]
   	   	  R[cbind(li2, li1)] <- -1
          num <- rep(0, m)
          for (i in 1:m){
               num[i] = sum(li1 == i) + sum(li2 == i)
          } 
          diag(R) <- num
          theta_HB = rowMeans(apply(rbind(beta, Sqsigmav, Sqsigma, lambda), MARGIN =  2, FUN = theta.HBSYC, y = y, X1 = X, R = R))
     }
     theta_HB
}
