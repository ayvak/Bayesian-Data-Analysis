data {
  int<lower=0> n;  // n is number of houses, d is number of features
  int<lower=0> d;
  int<lower=0> m; // m is number of feedback 
  int<lower=0> n_user; //  number of users
  matrix[n,d] dat;
  int<lower=0> f[m];
  int x[m];
  int xp[m];
  int u_index[m];
}

parameters {
  matrix[n_user,d] w;
  vector[d] mu;
}

transformed parameters{
  vector[m] a;
  vector[m] p1;
  vector[m] p2;
  for(i in 1:m){
    a[i]=(dat[x[i]]*w[u_index[i]]')-(dat[xp[i]]*w[u_index[i]]');
    p1[i] = normal_lcdf((a[i])|0 , 2);
    p2[i] = normal_lcdf((-1*a[i])| 0 ,2);
  }
}

model {
  for (j in 1:d)
    mu[j] ~ normal(0, 1);
    
  for (k in 1:n_user)
    for (j in 1:d)
      w[k,j] ~ normal(mu[j], 1);
      
  for (i in 1:m)
    target += log_sum_exp(p1[i]+bernoulli_lpmf(f[i]|0.95),p2[i]+bernoulli_lpmf(f[i]|0.05));  
}

generated quantities{
  vector[m] log_lik;
  for(i in 1:m)
    log_lik[i]= log_sum_exp(p1[i]+bernoulli_lpmf(f[i]|0.95),p2[i]+bernoulli_lpmf(f[i]|0.05));
}
