//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.

data {
  int<lower=0> n;  // n is number of houses, d is number of features
  int<lower=0> d;
  int<lower=0> m; // m is number of feedback 
  matrix[n,d] dat;
  int<lower=0> f[m];
  int x[m];
  int xp[m];
  
  //real pi;
}


parameters {
  //real sigma;
  simplex[d] w;
  
}
transformed parameters{
  vector[m] a;
  vector[m] p1;
  vector[m] p2;
  for(i in 1:m){
    a[i]=(dat[x[i]]*w)-(dat[xp[i]]*w);
    p1[i] = normal_lcdf((a[i])|0 , 1);
    p2[i] = normal_lcdf((-1*a[i])| 0 ,1);
  }
    
  
}



model {
  
  w~ dirichlet(rep_vector(1.0, d));
  for (i in 1:m)
    target += log_sum_exp(p1[i]+bernoulli_lpmf(f[i]|0.99),p2[i]+bernoulli_lpmf(f[i]|0.01));  
}

