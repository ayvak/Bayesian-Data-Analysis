

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
  vector[d] w;
  
}
transformed parameters{
  vector[m] a;
  vector[m] p1;
  vector[m] p2;
  for(i in 1:m){
    a[i]=(dat[x[i]]*w)-(dat[xp[i]]*w);
    p1[i] = normal_lcdf((a[i])|0 , 2);
    p2[i] = normal_lcdf((-1*a[i])| 0 ,2);
  }
}



model {
  for (j in 1:d)
    w[j] ~ normal(0, 1);
  for (i in 1:m)
    target += log_sum_exp(p1[i]+bernoulli_lpmf(f[i]|0.95),p2[i]+bernoulli_lpmf(f[i]|0.05));  
}

generated quantities{
  vector[m] log_lik;
  for(i in 1:m)
    log_lik[i]= log_sum_exp(p1[i]+bernoulli_lpmf(f[i]|0.95),p2[i]+bernoulli_lpmf(f[i]|0.05));
}
