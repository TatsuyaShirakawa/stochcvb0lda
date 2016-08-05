#pragma once

/**
 * Stochastic Collapsed Variational Bayes Inference with Zero-Order Approximation for Latent Dirichlet Allocation
 *
 *
 * -- Preliminary --
 *
 * k < {1 ... K}  : topics
 * w < {1 ... W}  : words
 * d < {1 ... D}  : documents
 * n < {1 ... Nd} : word-id in document d
 *
 * phi[k] : word distribution of topic k
 * theta[d] : topic proportions of document d
 * z[d,n] : (hard) assigned topic of the n-th word in a document d
 * gamma[d,k,n] : topic assignment probability for word-id n in document d
 * x[d,n] : n-th word in a document d
 * alpha : hyperparameter for prior distribution for theta[d]
 * beta : hyperparameter for prior distribution for phi[k]
 * 
 *
 * -- Generative Models --
 *
 * theta[d] ~ Dir(*|alpha)
 * phi[k] ~ Dir(*|beta)
 * z[d,n] ~ Mult(*|theta[d])
 * x[d,n] ~ Mult(*|phi[z[d,n]])
 *
 *
 * -- Algorithm Steps --
 *
 * 1. d <- document
 * 2. theta[d], gamma[d] <- infer(phi, d)
 * 3. phi <- update(phi, theta)
 *
 */

#include <vector>
#include <unordered_set>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include "dirichlet_distribution.hpp"

namespace bayes
{

  struct StochCVB0LDA
  {
  private:

    /**
     * Topic
     */
    struct Topic
    {
    private:
      std::vector<double> word_frequencies_;
      double scale_; // scale for word_frequencies
      double frequency_;

    public:

      Topic()
	:word_frequencies_(), scale_(1), frequency_(0)
      {}

      Topic(const std::vector<double>& word_frequencies)
	:word_frequencies_(), scale_(), frequency_()
      { initialize(word_frequencies); }

      Topic& operator=(const Topic& other)
      { 
	word_frequencies_ = other.word_frequencies_;
	scale_ = other.scale_;
	frequency_ = other.frequency_;
	return *this;
      }

    public:

      inline std::size_t size() const
      { return word_frequencies_.size(); }

      inline double word_frequency(std::size_t w) const
      { return word_frequencies_[w]*scale_; }

      inline double frequency() const
      { return frequency_; }

      inline double word_probability(std::size_t w) const
      { return word_frequency(w)/frequency(); }

    public:

      void initialize( const std::vector<double>& word_frequencies )
      {
	std::size_t W = word_frequencies.size();
	word_frequencies_ = word_frequencies;
	scale_ = 1;
	frequency_ = std::accumulate(word_frequencies_.begin(),
				     word_frequencies_.end(), 0.0);
      }

      void scale(double ratio)
      { 
	scale_ *= ratio; 
	frequency_ *= ratio;
      }
      
      void update_word_frequency(std::size_t w, double new_frequency)
      { 
	accumulate_word_frequency(w, new_frequency-word_frequency(w));
      }

      void accumulate_word_frequency(std::size_t w, double delta)
      { 
	word_frequencies_[w] += delta/scale_;
	frequency_ += delta;
      }
      
    }; // end of Topic


  public:
    typedef StochCVB0LDA self_type;
    typedef std::vector<std::pair<std::size_t, std::size_t> > bow_type;
  private:
    
    std::vector<double> alphas_;
    std::vector<double> betas_;    

    std::vector<Topic> topics_;

    uint64_t dcount_; // # of documents prevoiusly seen
    uint64_t ucount_; // # of previous updates

  private:
    // members for calculation efficiency
    double sum_betas_; 

  private:

      inline static double learning_rate(uint64_t t)
      { return pow( static_cast<double>(t) + 10, -0.75); }

  public:

    template <class RandomEngine>
    StochCVB0LDA(RandomEngine& engine,
		 const std::vector<double>& alphas,
		 const std::vector<double>& betas
		 )
      :alphas_(), betas_(),
       topics_(), 
       dcount_(), ucount_(),
       sum_betas_()
    { initialize(engine, alphas, betas); }

  public:
    template <class RandomEngine>    
    void initialize(RandomEngine& engine,
		    const std::vector<double>& alphas,
		    const std::vector<double>& betas
		    )
    {
      std::size_t K = alphas.size();
      std::size_t W = betas.size();
      alphas_ = alphas;
      betas_ = betas;
      sum_betas_ = std::accumulate(betas_.begin(), betas_.end(), 0.0);

      topics_.resize(K);
      dirichlet_distribution dirichlet;
      for(std::size_t k=0;k<num_topics();++k){
	std::vector<double> vals(dirichlet(betas_));
	topics_[k].initialize(dirichlet(engine, betas_));
      }

      dcount_ = 0;
      ucount_ = 0;
    }

  public:
    inline std::size_t num_topics() const
    { return alphas_.size(); }
    
    inline std::size_t num_vocabularies() const
    { return betas_.size(); }

  public:
    void save_alphas(std::vector<double>& alphas) const
    { alphas=alphas_; }

    void save_betas(std::vector<double>& betas) const
    { betas=betas_; }

  public:
    
    void save_word_probabilities(std::vector<std::vector<double> >& word_probs) const
    {
      word_probs.clear();
      word_probs.resize(num_topics(), std::vector<double>(num_vocabularies()));
      for(std::size_t k=0;k<num_topics();++k){
	for(std::size_t w=0;w<num_vocabularies();++w){
	  word_probs[k][w] = topics_[k].word_probability(w);
	}
      }
    }
    
    void save_word_frequencies(std::vector<std::vector<double> >& word_freqs) const
    {
      word_freqs.clear();
      word_freqs.resize(num_topics(), std::vector<double>(num_vocabularies()));
      for(std::size_t k=0;k<num_topics();++k){
	for(std::size_t w=0;w<num_vocabularies();++w){
	  word_freqs[k][w] = topics_[k].word_frequency(w);
	}
      }
    }
    
    /**
     * save scores of each word w in topic k 
     * ---------
     * word_score[k, w] 
     *   = phi[k, w] * log( phi[k, w] / prod(phi[k, w], k)^(1/K) )
     */
    void save_word_scores(std::vector<std::vector<double> >& result) const
    {
      result.clear();
      result.resize(num_topics(), std::vector<double>(num_vocabularies()));
      for(std::size_t w=0;w<num_vocabularies();++w){
	double avelog=0;
	for(std::size_t k=0;k<num_topics();++k){
	  double prob = topics_[k].word_probability(w);
	  if(prob == 0){
	    result[k][w] = 0;
	  }else{
	    avelog += log(prob);
	    result[k][w] = prob * log(prob);
	  }
	}
	avelog /= num_topics();
	for(std::size_t k=0;k<num_topics();++k){
	  double prob = topics_[k].word_probability(w);
	  result[k][w] -= prob * avelog;
	}
      }
    }
    
    
    void save_topic_frequencies(std::vector<double>& topic_freqs) const
    {	
      topic_freqs.resize(num_topics());
      for(std::size_t k=0;k<num_topics();++k){
	topic_freqs[k] = topics_[k].frequency();
      }
    }
    
    void save_topic_probabilities(std::vector<double>& topic_probs) const
    {
      topic_probs.resize(num_topics());
      double Z = 0;
      for(std::size_t k=0;k<num_topics();++k){
	topic_probs[k] = topics_[k].frequency();
	Z += topic_probs[k];
      }
      for(std::size_t k=0;k<num_topics();++k){
	topic_probs[k] /= Z;
      }
    }


  public:
    void infer(std::vector<double>& theta, // theta[k]
	       std::vector<std::vector<double> >& gamma, // gamma[k][i]
	       const bow_type& bow,
	       std::size_t maxiter = 200,
	       double thresh = 1e-5
	       ) const
    {
      // prepare
      std::vector<std::vector<double> > C(num_topics(), std::vector<double>(bow.size()));
      for(std::size_t k=0;k<num_topics();++k){
	for(std::size_t i=0;i<bow.size();++i){
	  std::size_t w = bow[i].first;
	  //	  std::size_t c = bow[i].second;
	  C[k][i] = 
	    (topics_[k].word_frequency(w) + betas_[w]) 
	    / (topics_[k].frequency() + sum_betas_);
	  assert(C[k][i] > 0);
	}
      }

      // initialize gamma
      gamma.clear();
      gamma.resize(num_topics(), std::vector<double>(bow.size()));
#if 1 
      //-- basic initialization
      for(std::size_t i=0;i<bow.size();++i){
	for(std::size_t k=0;k<num_topics();++k){
	  gamma[k][i] = 1.0/num_topics();
	}
      }
#else
      //-- greedy initialization
      std::vector<double> tmp(alphas_);
      double sum_topic_frequencies = 0;
      for(std::size_t k=0;k<num_topics();++k){
	sum_topic_frequencies += topics_[k].frequency();
      }
      std::size_t dsize = 0;
      for(std::size_t i=0;i<bow.size();++i){
	dsize += bow[i].second;
      }
      for(std::size_t k=0;k<num_topics();++k){
	tmp[k] = alpha_[k] + dsize * topics_[k].frequency() / sum_topic_frequencies;
      }
      for(std::size_t i=0;i<bow.size();++i){
	double V = 0;
	for(std::size_t k=0;k<num_topics();++k){
	  double v = tmp[k]*C[k][i];
	  gamma[k][i] = v;
	  V += v;
	}
	for(std::size_t k=0;k<num_topics();++k){
	  gamma[k][i] /= V;
	}
      }
#endif

      theta.clear();
      theta.resize(num_topics());
	
      // update 
      std::size_t iter_count=0;
      double change=0;

      std::vector<double> old_gamma(num_topics());
      std::vector<double> N(num_topics(), 0);
      for(iter_count=0;iter_count<maxiter;++iter_count){

	change=0;

	// 1. N[k] = sum(gamma[k,i], i) 
	for(std::size_t k=0;k<num_topics();++k){
	  N[k] = 0;
	  for(std::size_t i=0;i<bow.size();++i){
	    //	    std::size_t w = bow[i].first;
	    std::size_t c = bow[i].second;
	    double v = gamma[k][i] * c;
	    N[k] += v;
	  }
	}

	// 2. gamma[k,i] oc C[k][i] * (N[k]-gamma[k,i]+alpha_[k])
	for(std::size_t i=0;i<bow.size();++i){
	  //	  std::size_t w = bow[i].first;
	  //	  std::size_t c = bow[i].second;
	  double V = 0;
	  for(std::size_t k=0;k<num_topics();++k){
	    old_gamma[k] = gamma[k][i];
	    double v = C[k][i] * (N[k]-gamma[k][i]+alphas_[k]);
	    gamma[k][i] = v;
	    V += v;
	  }
	  assert(V>0);
	  for(std::size_t k=0;k<num_topics();++k){
	    gamma[k][i] /= V;
	    change += fabs(gamma[k][i] - old_gamma[k]);
	  }
	}

	if(change < thresh){ break; }

      } // end of update iteration

	// reculculate theta
	// E[theta[k]] oc (alpha[k]+N[k])

      double S = 0;
      for(std::size_t k=0;k<num_topics();++k){
	theta[k] = N[k] + alphas_[k];
	S += theta[k];
      }

      for(std::size_t k=0;k<num_topics();++k){
	theta[k] /= S;
      }
    } // end of infer

  public:

    void update(const std::vector<bow_type>& bows,
		std::size_t maxiter = 200,
		double thresh = 1e-5
		)
    {
      // increment counters and scale frequencies
      std::size_t prev_dcount=dcount_;
      dcount_ += bows.size();
      ++ucount_;
      double scale = 1;
      if(prev_dcount>0){
	scale = static_cast<double>(dcount_)/static_cast<double>(prev_dcount);
      }
      for(std::size_t k=0;k<num_topics();++k){
	topics_[k].scale(scale);
      }

      // infer
      std::vector<std::vector<double> > thetas(bows.size());
      std::vector<std::vector<std::vector<double> > > gammas(bows.size());
      std::ofstream ofs("theta.txt", std::ios::app|std::ios::out);
      for(std::size_t d=0;d<bows.size();++d){
	infer(thetas[d], gammas[d], bows[d], maxiter, thresh);
	for(std::size_t k=0;k<num_topics();++k){
	  if(k==0){
	    ofs << thetas[d][k];
	  }else{
	    ofs << "\t" << thetas[d][k];
	  }
	}
	ofs << std::endl;
      }
      ofs.close();

      // update phi
      double rate = learning_rate(ucount_);
      assert(rate > 0 );
      assert(rate <= 1 );
      scale = static_cast<double>(dcount_)/bows.size();
      double c = rate * scale;
      
      for(std::size_t k=0;k<num_topics();++k){
	topics_[k].scale(1-rate);
	// contributions from gammas
	std::unordered_set<std::size_t> ht;
	for(std::size_t d=0;d<bows.size();++d){
	  for(std::size_t i=0;i<bows[d].size();++i){
	    std::size_t w = bows[d][i].first;
	    ht.insert(w);
	    double v = c * gammas[d][k][i];
	    assert(v>=0);
	    topics_[k].accumulate_word_frequency(w, v);
	  }
	}
	// contributions from betas_
	for(const auto& w : ht){
	  double v = rate * betas_[w];
	  assert(v > 0);
	  topics_[k].accumulate_word_frequency(w, v);
	}
	
      }
#if 0
      // update alpha
      if(ucount_ >= 0){
	update_alphas(bows, thetas);
      }
#endif
//       std::cout << "alphas = [";
//       for(std::size_t k=0;k<num_topics();++k){
// 	std::cout << alphas_[k] << ", ";
//       }
//       std::cout << "]" << std::endl;


    } // end of update

    void update_alphas(const std::vector<bow_type>& bows,
		       const std::vector<std::vector<double> >& thetas,
		       double thresh=1e-5,
		       std::size_t maxitr=200
		       )
    {
      update_alphas_3(bows, thetas, thresh, maxitr);
    }

    /** update while keeping s = sum(alphas_[k],k) */
    void update_alphas_3(const std::vector<bow_type>& bows,
			 const std::vector<std::vector<double> >& thetas,
			 double thresh=1e-5,
			 std::size_t maxitr=200
			 )
    {
      std::size_t D = bows.size();
      std::size_t K = num_topics();

      double s = 0;
      for(std::size_t k=0;k<K;++k){
	s += alphas_[k];
      }

      std::vector<double> N(D, 0);
      std::vector<std::vector<double> > n(D, std::vector<double>(K, 0));
      for(std::size_t d=0;d<D;++d){
	// N[d]
	for(bow_type::const_iterator itr=bows[d].begin();itr!=bows[d].end();++itr){
	  N[d] += itr->second;
	}
	// compute n[d][k] considering theta oc alphas[k] + sum(gamma[k][i], i)
	for(std::size_t k=0;k<K;++k){
	  n[d][k] = std::max(thetas[d][k] * (s+N[d]) - alphas_[k], 0.0);
	}
      }

      std::vector<double> ave_log_points(K, 0);
      for(std::size_t d=0;d<D;++d){
	double sumtheta=0;
	for(std::size_t k=0;k<K;++k){
	  sumtheta += thetas[d][k];
	}
	for(std::size_t k=0;k<K;++k){
	  ave_log_points[k] += log(thetas[d][k]/sumtheta);
	}
      }
      for(std::size_t k=0;k<K;++k){
	ave_log_points[k] /= D;
      }

      /* update mean */

      std::vector<double> ms(K, 0);
      for(std::size_t k=0;k<K;++k){
	ms[k] = alphas_[k]/s;
      }
      
      std::vector<double> new_ms(ms);

      double err=0;
      std::size_t count=0;
      
      while(count < maxitr){
	err = 0;
	++count;

	double V=0;
	for(std::size_t k=0;k<K;++k){
	  V += new_ms[k]*(ave_log_points[k]-math::digamma(s*new_ms[k]));
	}
	double alpha0=1e-8;
	std::vector<double> tmp_alphas(K,0);
	double sum_tmp_alphas = 0;
	for(std::size_t k=0;k<K;++k){
	  tmp_alphas[k] = math::inverse_digamma(ave_log_points[k] - V);
	  if(tmp_alphas[k]<=0){tmp_alphas[k]=alpha0;}
	  sum_tmp_alphas += tmp_alphas[k];
	}
	for(std::size_t k=0;k<K;++k){
	  double m = tmp_alphas[k]/sum_tmp_alphas;
	  err += fabs(m-new_ms[k]);
	  new_ms[k] = m;
	}

	if(err < thresh){ break; }
      }


      /* update precision */
      // this computation is not suitable for online computation,
      // so we do batch computatin
      count = 0;
      double new_s = s;
#if 0
      while(count < maxitr){
	err = 0;
	++count;
	
	double g = D*math::digamma(new_s);
	double H = D*math::trigamma(new_s);
	for(std::size_t k=0;k<K;++k){
	  double m = new_ms[k];
	  g -= D*m*math::digamma(new_s*m);
	  H -= D*m*m*math::trigamma(new_s*m);
	  for(std::size_t d=0;d<D;++d){
	    g += m*log(thetas[d][k]);
	  }
	}

	double s0 = 1e-8;
	double v = 1/(1/new_s + (1/(new_s*new_s))*g/H);
	if(v<=0){ v = s0; }
	err = fabs(new_s-v);
	new_s = v;
	if(err < thresh){ break; }
      }
#endif

      double rate = learning_rate(ucount_);


      // update alpha
      for(std::size_t k=0;k<K;++k){
	alphas_[k] = (1-rate)*alphas_[k] + rate*new_s*new_ms[k];
      }

    }

    void update_alphas_2(const std::vector<bow_type>& bows,
		       const std::vector<std::vector<double> >& thetas,
		       double thresh=1e-5,
		       std::size_t maxitr=200
		       )
    {
      std::size_t D = bows.size();
      std::size_t K = num_topics();

      for(std::size_t d=0;d<D;++d){
	std::cout << "theta[" << d << "] = [";
	for(std::size_t k=0;k<K;++k){
	  std::cout << thetas[d][k] << ", ";
	}
	std::cout << "]" << std::endl;
      }

      // sum_alphas = sum(alphas_[i], i);
      double sum_alphas = 0;
      for(std::vector<double>::const_iterator itr=alphas_.begin();itr!=alphas_.end();++itr){
	sum_alphas += *itr;
      }

      // N[d] = number of words in the document d
      // n[d][k] = number of words assigned to the topic k in the document d
      std::vector<double> N(D, 0);
      std::vector<std::vector<double> > n(D, std::vector<double>(K, 0));
      for(std::size_t d=0;d<D;++d){
	// N[d]
	for(bow_type::const_iterator itr=bows[d].begin();itr!=bows[d].end();++itr){
	  N[d] += itr->second;
	}
	std::cout << "N[" << d << "] = " << N[d] << std::endl;
	// compute n[d][k] by theta oc alphas[k] + sum(gamma[k][i], i)
	for(std::size_t k=0;k<K;++k){
	  n[d][k] = std::max(thetas[d][k] * (sum_alphas+N[d]) - alphas_[k], 0.0);
	  std::cout << "n[" << d << "][" << k << "] = " << n[d][k] << std::endl;
	}
      }

      // log probability
      double logprob=0;
      for(std::size_t d=0;d<D;++d){
	logprob += math::loggamma(sum_alphas) - math::loggamma(sum_alphas+N[d]);
	for(std::size_t k=0;k<K;++k){
	  logprob += math::loggamma(alphas_[k]+n[d][k]) - math::loggamma(alphas_[k]);
	}
      }

      double err=0;
      std::size_t count=0;

      std::vector<double> new_alphas(alphas_);
      std::vector<double> new_alphas_save(alphas_);
      double new_sum_alphas=sum_alphas;
      while(count < maxitr){

	err = 0;
	++count;

	double Z=0;
	for(std::size_t d=0;d<D;++d){
	  Z += math::digamma(N[d]+new_sum_alphas);
	}
	Z -= D*math::digamma(new_sum_alphas);

	new_alphas_save = new_alphas;
	double alpha0 = 1e-8;
	for(std::size_t k=0;k<K;++k){
	  double a = new_alphas[k];
	  new_alphas[k] = 0;
	  for(std::size_t d=0;d<D;++d){
	    new_alphas[k] += math::digamma(n[d][k]+a);
	  }
	  new_alphas[k] -= D*math::digamma(a);
	  new_alphas[k] *= (a/Z);
	  if(new_alphas[k] <= 0){ new_alphas[k] = alpha0; }
	  err += fabs(new_alphas[k]-a);
	}
	
	// check if log probability increased
	double new_logprob=0;
	for(std::size_t d=0;d<D;++d){
	  //	  std::cout << "--\t" << new_sum_alphas << "\t" << N[d] << std::endl;
	  new_logprob += math::loggamma(new_sum_alphas) - math::loggamma(new_sum_alphas+N[d]);
	  for(std::size_t k=0;k<K;++k){
	    //	    std::cout << "@k\t" << new_alphas[k]  << "\t" << n[d][k] << std::endl;
	    new_logprob += math::loggamma(new_alphas[k]+n[d][k]) - math::loggamma(new_alphas[k]);
	  }
	}

	if(new_logprob <= logprob){
	  new_alphas = new_alphas_save;
	  break;
	}else{
	  // sum_alphas = sum(alphas_[i], i);
	  new_sum_alphas = 0;
	  for(std::vector<double>::const_iterator itr=new_alphas.begin();itr!=new_alphas.end();++itr){
	    new_sum_alphas += *itr;
	  }
	  if(err < thresh){ break; }
	}
      }
      double rate = learning_rate(ucount_);


      // update alpha
      std::cout << "new_alphas = [";
      for(std::size_t k=0;k<K;++k){
	std::cout << new_alphas[k] << ", ";
	alphas_[k] = (1-rate)*alphas_[k] + rate*new_alphas[k];
      }
      std::cout << "]" << std::endl;
      exit(0);
    }

    void update_alphas_1(const std::vector<bow_type>& bows,
			   const std::vector<std::vector<double> >& thetas,
			   double thresh=1e-5,
			   std::size_t maxitr=200
			   )
    {
      std::size_t D = bows.size();
      std::size_t K = num_topics();

      // sum_alphas = sum(alphas_[i], i);
      double sum_alphas = 0;
      for(std::vector<double>::const_iterator itr=alphas_.begin();itr!=alphas_.end();++itr){
	sum_alphas += *itr;
      }
	
      // N[d] = number of words in the document d
      // n[d][k] = number of words assigned to the topic k in the document d
      std::vector<double> N(D, 0);
      std::vector<std::vector<double> > n(D, std::vector<double>(K, 0));
      for(std::size_t d=0;d<D;++d){
	// N[d]
	for(bow_type::const_iterator itr=bows[d].begin();itr!=bows[d].end();++itr){
	  N[d] += itr->second;
	}
	// n[d][k] は theta oc alphas[k] + sum(gamma[k][i], i) を使って計算
	for(std::size_t k=0;k<K;++k){
	  n[d][k] = std::max(thetas[d][k] * (sum_alphas+N[d]) - alphas_[k], 0.0);
	}
      }

      double err=0;
      std::size_t count=0;

       std::vector<double> new_alphas(alphas_);
//       std::vector<double> new_alphas(num_topics(), 0);

      // initialize new_alphas
//       for(std::size_t k=0;k<K;++k){
// 	new_alphas[k] = 0;
// 	for(std::size_t d=0;d<D;++d){
// 	  new_alphas[k] += static_cast<double>(n[d][k])/N[d];
// 	}
// 	new_alphas[k] /= D;
//       }

      // Newton-Raphson Update
      while(count < maxitr){
	err = 0;
	++count;

	// sum_alphas = sum(alphas[k], k)
	sum_alphas = 0;
	for(std::vector<double>::const_iterator itr=new_alphas.begin();itr!=new_alphas.end();++itr){
	  sum_alphas += *itr;
	}
	
	// c
	double c = D*math::trigamma(sum_alphas);
	for(std::size_t d=0;d<D;++d){
	  c -= math::trigamma(sum_alphas+N[d]);
	}
	
	// g, q
	std::vector<double> g(K, 0);
	std::vector<double> q(K, 0);
	for(std::size_t k=0;k<K;++k){
	  g[k] = D*(math::digamma(sum_alphas)-math::digamma(new_alphas[k]));
	  q[k] = -(D*math::trigamma(new_alphas[k])); 
	  for(std::size_t d=0;d<D;++d){
	    g[k] -= math::digamma(sum_alphas+N[d]) - math::digamma(new_alphas[k]+n[d][k]);
	    q[k] += math::trigamma(new_alphas[k]+n[d][k]);
	  }
	}
	
	// b
	double b1=0;
	double b2=1.0/c;
	for(std::size_t k=0;k<K;++k){
	  b1 += g[k]/q[k];
	  b2 += 1.0/q[k];
	}
	double b = b1/b2;

	double alpha0 = 1e-8;
	for(std::size_t k=0;k<K;++k){
	  double a = new_alphas[k];
	  new_alphas[k] -= (g[k]-b)/q[k];
	  if(new_alphas[k] <= 0){ new_alphas[k] = alpha0; }
	  err += fabs(a-new_alphas[k]);
	}
	if(err < thresh){ break; }
      }

      double rate = learning_rate(ucount_);
      for(std::size_t k=0;k<K;++k){
	alphas_[k] = (1-rate)*alphas_[k] + rate*new_alphas[k];
      }
    }
    
  public:
    double perplexity(const std::vector<bow_type>& bows,
		      std::size_t maxiter=200,
		      double thresh=1e-5
		      ) const
    {
      // infer gamma
      std::vector<std::vector<double> > thetas(bows.size());
      std::vector<std::vector<std::vector<double> > > gammas(bows.size());
      for(std::size_t d=0;d<bows.size();++d){
	infer(thetas[d], gammas[d], bows[d], maxiter, thresh);
      }

      // log_likelihood
      std::size_t N=0;
      double log_lik = 0;
      for(std::size_t d=0;d<bows.size();++d){
	for(std::size_t i=0;i<bows[d].size();++i){
	  std::size_t w = bows[d][i].first;
	  std::size_t c = bows[d][i].second;
	  double prob = 0;
	  for(std::size_t k=0;k<num_topics();++k){
	    prob += thetas[d][k] * topics_[k].word_probability(w);
	  }
	  log_lik += log(prob) * c;
	  N += c;
	}
	  
      }
      return exp(-log_lik/N);

    } // end of perplexity
    
  }; // end of StochCVB0LDA

} // end of bayes

