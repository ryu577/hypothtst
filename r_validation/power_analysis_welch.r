n1=10; n2=10;  sg1=400; sg2=400;  dlt=1000
pv = replicate(10^4,
       t.test(rnorm(n1,30927.213013013,sg1),rnorm(n2,30927.213013013-dlt,sg2))$p.val)
mean(pv <= .05)


ssize.welch = function(alpha=0.05, power=0.90, mu1, mu2, sigma1, sigma2, n2n1r, use_exact=FALSE)
 {
  mud<-mu1-mu2
  sigsq1<-sigma1^2
  sigsq2<-sigma2^2

  numint<-1000
  dd<-0.00001
  coevec<-c(1,rep(c(4,2),numint/2-1),4,1)
  intl<-(1-2*dd)/numint
  bvec<-intl*seq(0,numint)+dd

  #Z method
  za<-qnorm(1-alpha/2)
  zb<-qnorm(power)
  n1<-ceiling(((sigsq1+sigsq2/n2n1r)*(za+zb)^2)/(mud^2))
  n2<-ceiling(n1*n2n1r)

  if(use_exact) #Exact method
   {
    n1 = n1-1
    n2 = n2-1
    powere<-0

    while(powere<power)
     {
      n1<-n1+1
      n2<-ceiling(n1*n2n1r)
      sigsqt<-sigsq1/n1+sigsq2/n2
      hsigsqt<-sqrt(sigsqt)
      wpdf<-(intl/3)*coevec*dbeta(bvec,(n1-1)/2,(n2-1)/2)
      dft<-n1+n2-2 
      p1<-(n1-1)/dft
      p2<-1-p1
      s1<-sigsq1/n1
      s2<-sigsq2/n2
      b12<-(s1/p1)*bvec+(s2/p2)*(1-bvec)
      r1<-(s1/p1)*bvec/b12
      r2<-1-r1
      dfevec<-1/((r1^2)/(n1-1)+(r2^2)/(n2-1))
      tdfea<-qt(1-alpha/2,dfevec)
      powere<-sum(wpdf*pt(-tdfea*sqrt(b12/sigsqt),dft,mud/hsigsqt))+1-sum(wpdf*pt(tdfea*sqrt(b12/sigsqt),dft,mud/hsigsqt))
      }
    }

  c(n1=n1,n2=n2)
  }
