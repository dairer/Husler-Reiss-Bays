functions{

  // density of gev
  real gev_likelihood(real z, real loc, real shape, real scale){

    real tm;

    if(scale <= 0)
      reject(scale)

    if((shape < 0) && (z > (loc - (scale/shape))))
      reject(loc, scale, shape)

    if((shape > 0) && (z < (loc - (scale/shape))))
          reject(loc, scale, shape)

    if(shape != 0){
      tm = (1 + shape*((z - loc)/scale))^(-1/shape);
    }else{
      tm = exp(-(z - loc)/scale);
    }

    return ((1/scale)*(tm^(shape+1))*exp(-tm));
  }

  // gev cdf
  real gev_distribution(real z, real loc, real shape, real scale){
    real tm;

    if(scale <= 0)
      reject(scale)

    if((shape < 0) && (z > (loc - (scale/shape))))
      reject(loc, scale, shape)

    if((shape > 0) && (z < (loc - (scale/shape))))
        reject(loc, scale, shape)

    if(shape != 0){
      if((1 + shape*((z - loc)/scale)) > 0){
        tm = (1 + shape*((z - loc)/scale))^(-1/shape);
      }
      else{
        tm = 0^(-1/shape);
      }
    }else{
      tm = exp(-(z - loc)/scale);
    }

    return(exp(-tm));
  }

    // likelihood function of Hüsler-Reiss bivariate copula
  real HuslerReiss(real x, real y, real loc_1, real loc_2, real scale_1,
                   real scale_2, real shape_1, real shape_2, real lambda){

      real u;
      real v;
      real density_marg_1;
      real density_marg_2;
      real z;
      real a;
      real density_HR;
      real LL;


      if(lambda<=0){
        reject(lambda);
      }

      if(scale_1 <= 0)
        reject(loc_1, scale_1, shape_1)

      if((shape_1 < 0) && (x > (loc_1 - (scale_1/shape_1))))
        reject(loc_1, scale_1, shape_1)

      if((shape_1 > 0) && (x < (loc_1 - (scale_1/shape_1))))
            reject(loc_1, scale_1, shape_1)

      if(scale_2 <= 0)
        reject(loc_2, scale_2, shape_2)

      if((shape_2 < 0) && (y > (loc_2 - (scale_2/shape_2))))
        reject(loc_2, scale_2, shape_2)

      if((shape_2 > 0) && (y < (loc_2 - (scale_2/shape_2))))
            reject(loc_2, scale_2, shape_2)

      // likelihood margin 1
      density_marg_1 = gev_likelihood(x, loc_1, shape_1, scale_1);
      if(density_marg_1 < 0 || density_marg_2 >=1){
        density_marg_1 = 0;
      }

      // likelihood margin 2
      density_marg_2 = gev_likelihood(y, loc_2, shape_2, scale_2);
      if(density_marg_2 <= 0 || density_marg_2 >=1){
          density_marg_2 = 0;
      }


      if(density_marg_2 > 0 && density_marg_1 > 0){

      // transform data to uniform
      u = gev_distribution(x, loc_1, shape_1, scale_1);
      v = gev_distribution(y, loc_2, shape_2, scale_2);

      z = log(log(u) / log(v));
      a = (1/lambda) + (lambda/2)*z;


      density_HR = (1 / (u * v))*(exp(log(u) * normal_cdf(a, 0, 1) +
      log(v) * normal_cdf((1/lambda) + (lambda/2) * log(log(v) / log(u)), 0, 1)) *
      (normal_cdf(1/lambda - (lambda/2)*z, 0, 1) * normal_cdf(a, 0, 1) +
      (lambda/2) * -1 / log(v) * exp(-0.5 * a^2) / sqrt(2 * pi())));


      LL = log(density_HR*density_marg_1*density_marg_2);
        return(LL);
      }else{
        return(negative_infinity());
      }
    }
}

data {
  int <lower = 0> N;
  real x[N];
  real y[N];
  real <lower=0> prior_mean;
  real <lower=0> prior_sd;
  real <lower=0>prior_mean_sig_1;
  real <lower=0>prior_sd_sig_1;
  real prior_mean_loc_1;
  real <lower=0>prior_sd_loc_1;
  real shape_lower_1;
  real shape_upper_1;
  real <lower=0>prior_mean_sig_2;
  real <lower=0>prior_sd_sig_2;
  real prior_mean_loc_2;
  real <lower=0>prior_sd_loc_2;
  real shape_lower_2;
  real shape_upper_2;
  }


transformed data {
  real maxObs_1 = max(x);
  real maxObs_2 = max(y);

  real minObs_1 = min(x);
  real minObs_2 = min(y);

}

parameters {
  real <lower=0, upper = 10> lambda;
  real <lower=0> scale_1;
  real <lower=0> scale_2;
  real <lower=-0.5, upper = 0.5> shape_1;
  real <lower=-0.5, upper = 0.5> shape_2;

   real<lower=(shape_1 > 0 ? minObs_1 : negative_infinity()),
       upper=(shape_1 > 0 ? positive_infinity() : maxObs_1 )> loc_1;

   real<lower=(shape_2 > 0 ? minObs_2 : negative_infinity()),
       upper=(shape_2 > 0 ? positive_infinity() : maxObs_2 )> loc_2;
}

model {
  lambda ~ normal(prior_mean,prior_sd); // prior on HR dependence parameter

  loc_1 ~ normal(prior_mean_loc_1,prior_sd_loc_1);
  shape_1 ~ uniform(shape_lower_1,shape_upper_1)T[shape_lower_1,shape_upper_1];
  scale_1 ~ normal(prior_mean_sig_1,prior_sd_sig_1);

  loc_2 ~ normal(prior_mean_loc_2,prior_sd_loc_2);
  shape_2 ~ uniform(shape_lower_2,shape_upper_2)T[shape_lower_2,shape_upper_2];
  scale_2 ~ normal(prior_mean_sig_2,prior_sd_sig_2);

  for(n in 1:N){
    target += HuslerReiss(x[n], y[n], loc_1, loc_2, scale_1,scale_2,shape_1,shape_2,lambda);
  }
}
