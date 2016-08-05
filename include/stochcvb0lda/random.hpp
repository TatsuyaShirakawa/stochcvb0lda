#pragma once

#include <random>

namespace bayes{

  struct DirichletDistribution
  {
    DirichletDistribution()

(constructor)
 
constructs new distribution 
(public member function)
reset
 
resets the internal state of the distribution 
(public member function)
Generation
operator()
 
generates the next random number in the distribution 
(public member function)
Characteristics
alpha
beta
 
returns the distribution parameters 
(public member function)
param
 
gets or sets the distribution parameter object 
(public member function)
min
 
returns the minimum potentially generated value 
(public member function)
max
 
returns the maximum potentially generated value 
(public member function)
Non-member functions
operator==
operator!=
 
compares two distribution objects 
(function)
operator<<
operator>>
 
performs stream input and output on pseudo-random number distribution 
(function template)
  }; // end of DirichletDistribution

} // end of bayes
