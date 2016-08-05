//#include <falco/lda/stochastic_cvb0_lda.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#define PRINT_TOPICS_DURING_ITERATION
#define PRINT_PERPLEXITY_DURING_ITERATION

const size_t MAXLINE = 8192;

typedef std::vector<std::pair<size_t, size_t> > bow_type;

namespace{

  /* mapping word -> word_id*/
  struct VocabularyMap
  {
    std::unordered_map<std::string, size_t> w2n_;
    std::vector<std::string> n2w_;

  public:
    VocabularyMap():w2n_(),n2w_(){}
  public:

    size_t entry(const std::string& w)
    {
      if(auto (itr = w2n_.find(w)) != w2n_.end() ){
	return *itr;
      }else{
	size_t n = w2n_.size();
	w2n_.insert(std::make_pair(w, n));
	n2w_.push_back(w);
	return n;
      }
    }

    inline size_t number(const std::string& w) const
    { return w2n_[w]; }

    inline const std::string& unnumber(size_t n) const
    { return n2w_[n]; }

    inline size_t size() const
    { return w2n_.size(); }
  };


  struct Config
  {
    Config(): bow_file(), voc_file(),
	      num_topics(), batch_size(), perp_sample_size(),
	      every_perp_time(), every_dump_time()
    {}
    
    std::string bow_file;
    std::string voc_file;
    std::size_t num_topics;
    std::size_t batch_size;
    std::size_t perp_sample_size;
    std::size_t every_perp_time;
    std::size_t every_dump_time;
  }

/* load arguments */
falco::OptionMap parse_args(int narg, char** argv)
{
  falco::CommandLineOption opt;
  opt.add_option("-b", "--bow_file", "bow_file", "bag-of-words file (ref. UCI Machine Learning Repository's Bag of Words Data Set");
  opt.add_option("-v", "--voc_file", "voc_file", "vocabulary file (ref. UCI Machine Learning Repository's Bag of Words Data Set");
  opt.add_option("-k", "--k", "num_topics", "number of topics");
  opt.add_option("-mb", "--minibatch_size", "minibatch_size", "minibatch size");
  opt.add_option("-mi", "--max_iteration", "max_iteration", "max iteration");
  opt.add_option("-ps", "--perp_sample_size", "perp_sample_size", "sample size for calculating perplexity");
  opt.add_option("-pt", "--perp_time_span", "perp_time_span", "time span for calculating perplexity");
  opt.add_option("-dt", "--dump_time_span", "dump_time_span", "time span for dumping perplexity");

  falco::OptionMap result = opt.parse(narg, argv);

  if(!result["bow_file"].option_used()
     ||!result["voc_file"].option_used()
     ||!result["num_topics"].option_used()
     ||!result["minibatch_size"].option_used()
     ||!result["max_iteration"].option_used()
     ){
    std::cerr << "invalide arguments." << std::endl;
    opt.show_help();
    exit(1);
  }

  if(!result["perp_sample_size"].option_used()){
    result["perp_sample_size"] = "1000";
  }
  if(!result["perp_time_span"].option_used()){
    result["perp_time_span"] = "1000";
  }
  if(!result["dump_time_span"].option_used()){
    result["dump_time_span"] = "10000";
  }

  return result;
}

void _read_bow_file(const std::string& bow_file,
		   std::vector<bow_type>& bows,
		   size_t& num_documents,
		   size_t& num_vocabularies,
		   size_t& total_bow_size
		   )
{
  std::ifstream ifs(bow_file.c_str());
  if(!ifs || !ifs.is_open()){
    std::cerr << "file \"" << bow_file << "\" cannot open." << std::endl;
    exit(1);
  }

  char line[MAXLINE];

  // number of documents
  ifs.getline(line, MAXLINE);
  num_documents = atoi(line);
  
  // vocabulary size
  ifs.getline(line, MAXLINE);
  num_vocabularies = atoi(line);
  
  // total words
  ifs.getline(line, MAXLINE);
  total_bow_size = atoi(line);

  bows.clear();
  bows.resize(num_documents);
  for(size_t i=0;i<total_bow_size;++i){
    ifs.getline(line, MAXLINE);
    size_t doc_no = atoi(line) - 1;
    size_t pos = 0;
    do{++pos;}while(line[pos]!=' ');
    size_t voc_no = atoi(&line[pos]) - 1;
    ++pos;
    do{++pos;}while(line[pos]!=' ');
    size_t count = atoi(&line[pos]);
    bows[doc_no].push_back(std::pair<size_t, size_t>(voc_no, count));
  }

  ifs.close();

  std::cout << "reading bag-of-words file ... ok" << std::endl;
  std::cout << std::endl;

  std::cout << "* bag-of-words file (" << bow_file << ")" << std::endl;
  std::cout << "\t- number of documents: " << num_documents << std::endl;
  std::cout << "\t- number of vocabularies: " << num_vocabularies << std::endl;
  std::cout << "\t- total size of bag-of-words: " << total_bow_size << std::endl;
  std::cout << std::endl;

}

void _read_voc_file(const std::string& voc_file,
		    size_t& num_vocabularies,
		    std::vector<std::string>& vocs
		    )
{
  std::ifstream ifs(voc_file.c_str());
  if(!ifs || !ifs.is_open()){
    std::cerr << "file \"" << voc_file << "\" cannot open." << std::endl;
    exit(1);
  }
  
  vocs.clear();
  vocs.resize(num_vocabularies);
  //  char line[MAXLINE];
  for(size_t i=0;i<num_vocabularies;++i){
    ifs >> vocs[i];
  }
  std::cout << "reading vocabularies file ... ok" << std::endl;
  std::cout << std::endl;
}

void read_files(const std::string& bow_file,
		const std::string& voc_file,
		std::vector<bow_type>& bows,
		std::vector<std::string>& vocs,
		size_t& num_documents,
		size_t& num_vocabularies,
		size_t& total_bow_size
		)
{
  _read_bow_file(bow_file, bows, num_documents, num_vocabularies, total_bow_size);
  _read_voc_file(voc_file, num_vocabularies, vocs);
}


struct compare_t
{
public:
  inline bool operator()(const std::pair<size_t, std::pair<double, double> >& e1,
			 const std::pair<size_t, std::pair<double, double> >& e2
			 )
  { return (e1.second.first >= e2.second.first)
      || ( (e1.second.first == e2.second.first) && (e1.first <= e2.first) );
  }
};

struct topic_word_t
{
  size_t word_id;
  size_t topic_id;
  std::string word;
  double score;
  double prob;
  double freq;
public:
  topic_word_t()
    :word_id(), topic_id(), word(), score(), prob(), freq()
  {}
  topic_word_t(size_t word_id, size_t topic_id
	       , const std::string& word
	       , double score ,double prob, double freq
	       )
    :word_id(word_id), topic_id(topic_id), word(word), score(score), prob(prob), freq(freq)
  {}
public:
  inline bool operator<(const topic_word_t& other) const
  {
    if(this->score < other.score){
      return true;
    }else{
      return (this->score == other.score) && (this->word_id < other.word_id);
    }
  }
};

struct pred_topic_word
{
public:
  inline bool operator()(const topic_word_t& w1, const topic_word_t& w2) const
  { return !(w1<w2); }
};

void show_topics(const falco::lda::StochCVB0LDA& lda,
		 const VocabularyMap& vocmap,
		 size_t N=50
		 )
{

  std::vector<std::vector<double> > word_dists;
  std::vector<std::vector<double> > word_freqs;
  std::vector<std::vector<double> > word_scores;
  std::vector<double> topic_freqs;
  std::vector<double> topic_dists;
  
  lda.save_word_dists(word_dists);
  lda.save_word_freqs(word_freqs);
  lda.save_topic_dists(topic_dists);
  lda.save_topic_freqs(topic_freqs);
  lda.save_word_scores(word_scores);

  std::vector<std::vector<topic_word_t> > topic_words(lda.num_topics(), std::vector<topic_word_t>(lda.num_vocabularies()));
  for(size_t k=0;k<lda.num_topics();++k){
    for(size_t w=0;w<lda.num_vocabularies();++w){
      topic_words[k][w] = topic_word_t(w, k,
				       vocmap.unnumber(w),
				       word_scores[k][w],
				       word_dists[k][w],
				       word_freqs[k][w]
				       );
    }
  }
  

  std::cout << std::endl;
  std::cout << "\t" << "== summary of topics ==" << std::endl;
  std::cout << std::endl;
  std::cout << std::left << "topic" 
	    << "\t" << std::left << std::setw(10) << "prob."
	    << "\t" << std::left << std::setw(10) << "freq."
	    << std::endl;
  std::cout << std::left << "-----" 
	    << "\t" << std::left << std::setw(10) << "-----"
	    << "\t" << std::left << std::setw(10) << "-----"
	    << std::endl;
  for(size_t k=0;k<lda.num_topics();++k){
    std::cout << std::left << k 
	      << "\t" << std::left << std::setw(10) << topic_dists[k]
	      << "\t" << std::left << std::setw(10) << topic_freqs[k]
	      << std::endl;
  }

  std::cout << std::endl;
  std::cout << "\t" << "== detail of topics ==" << std::endl;
  std::cout << std::endl;
  for(size_t k=0;k<lda.num_topics();++k){
    std::cout << "topic: " << k
	      << "\tprob.: " << topic_dists[k] 
	      << "\tfreq.: " << topic_freqs[k]
	      << std::endl;
    std::cout << std::endl;
    std::cout << std::left << "rank" 
	      << "\t" << std::left << std::setw(10) << "score"
	      << "\t" << std::left << std::setw(10) << "prob."
	      << "\t" << std::left << std::setw(10) << "freq."
	      << "\t" << std::left << std::setw(10) << "word"
	      << "\t" << std::endl;
    std::cout << std::left << "----" 
	      << "\t" << std::left << std::setw(10) << "-----"
	      << "\t" << std::left << std::setw(10) << "-----"
	      << "\t" << std::left << std::setw(10) << "-----"
	      << "\t" << std::left << std::setw(10) << "----"
	      << "\t" << std::endl;
    std::sort(topic_words[k].begin(), topic_words[k].end(), pred_topic_word());
    for(size_t n=0; (n<N && n<lda.num_vocabularies()); ++n){
      topic_word_t w = topic_words[k][n];
      std::cout << std::left << n+1
		<< "\t" << std::left << std::setw(10) << w.score
		<< "\t" << std::left << std::setw(10) << w.prob
		<< "\t" << std::left << std::setw(10) << w.freq
		<< "\t" << std::left << std::setw(10) << w.word
		<< std::endl;
    }
    std::cout << std::endl;
  }
}


int main(int narg, char** argv)
{
  /* command-line arguments */
  falco::OptionMap optmap = parse_args(narg, argv);
  std::string bow_file = optmap["bow_file"].as<std::string>();
  std::string voc_file = optmap["voc_file"].as<std::string>();
  size_t num_topics = optmap["num_topics"].as<size_t>();
  size_t minibatch_size = optmap["minibatch_size"].as<size_t>();
  size_t max_iteration = optmap["max_iteration"].as<size_t>();
  size_t perp_sample_size = optmap["perp_sample_size"].as<size_t>();
  size_t perp_time_span = optmap["perp_time_span"].as<size_t>();
  size_t dump_time_span = optmap["dump_time_span"].as<size_t>();

  std::cout << "parsing input arguments ... ok" << std::endl;
  std::cout << std::endl;

  std::cout << "* Input Arguments:" << std::endl;
  std::cout << "\t- bag-of-words file: " << bow_file << std::endl;
  std::cout << "\t- vocabularies file: " << voc_file << std::endl;
  std::cout << "\t- number of topics: " << num_topics << std::endl;
  std::cout << "\t- minibatch size: " << minibatch_size << std::endl;
  std::cout << "\t- max iteration: " << max_iteration << std::endl;
  std::cout << "\t- perp sample size: " << perp_sample_size << std::endl;
  std::cout << "\t- perp time span: " << perp_time_span << std::endl;  
  std::cout << "\t- dump time span: " << dump_time_span << std::endl;  
  std::cout << std::endl;


  /* read flies */
  std::vector<bow_type> bows;
  std::vector<std::string> vocs;
  size_t num_documents;
  size_t num_vocabularies;
  size_t total_bow_size;
  read_files(bow_file, voc_file, bows, vocs, num_documents, num_vocabularies, total_bow_size);

  /* create mappings between vocaburaries and ids */
  VocabularyMap vocmap;
  for(size_t i=0;i<num_vocabularies;++i){
    vocmap.entry(vocs[i]);
  }
  std::cout << "making vocabulary map ... ok" << std::endl;
  std::cout << std::endl;
  

  /* initialize */
  //  std::vector<double> alphas(num_topics, 0.5);
  std::vector<double> alphas(num_topics);
  falco::UniformGenerator<double> udgen;
  double alpha_lb = 1.0/(50*num_topics)*0.5;
  double alpha_ub = 1.0/(50*num_topics)*1.5;
  for(size_t k=0;k<num_topics;++k){
    alphas[k] = udgen(alpha_lb, alpha_ub);
    assert(alphas[k] > 0);
  }
  //  std::vector<double> etas(num_vocabularies, 0.5);
  std::vector<double> betas(num_vocabularies);
  for(size_t w=0;w<num_vocabularies;++w){
    //    etas[w] = udgen(0.1, 0.3);
    betas[w] = 0.01;
  }
  falco::lda::StochCVB0LDA lda(alphas, betas);
  std::cout << "initialize algorithm ... ok" << std::endl;
  std::cout << std::endl;

  /* update */

  size_t max_update_iter = 50;
  double thresh = 1e-5;

  size_t max_perp_iter = 50;
  double perp_thresh = 1e-5;

  size_t last_max_perp_iter = 200;
  double last_perp_thresh = 1e-5;

  size_t seed = 0;

  std::vector<bow_type> mb(minibatch_size); 
  falco::UniformGenerator<size_t> ugen;
  for(size_t it=0;it<max_iteration;++it){
    for(size_t i=0;i<minibatch_size;++i){
      size_t n;
      do{n = ugen(0, num_documents);}while(bows[n].size()==0);
      mb[i] = bows[n];
    }
    lda.update(mb, max_update_iter, thresh);
    if((minibatch_size * it)/100 < (minibatch_size * (it+1))/100){
      std::cout << "[" << minibatch_size*(it+1) << "]" << std::endl;
    }
    

#ifdef PRINT_TOPICS_DURING_ITERATION
    if(dump_time_span > 0){
      if((it +1) != max_iteration
	 && (minibatch_size * it)/dump_time_span < (minibatch_size * (it+1))/dump_time_span
	 ){
	show_topics(lda, vocmap, 50);
      }
    }
#endif

#ifdef PRINT_PERPLEXITY_DURING_ITERATION
    // perplexity
    if(perp_sample_size > 0){
      if((it+1) != max_iteration
	 && (minibatch_size * it)/perp_time_span < (minibatch_size * (it+1))/perp_time_span
	 ){
	std::vector<bow_type> perp_sample(perp_sample_size); 
	for(size_t j=0;j<perp_sample_size;++j){
	  size_t n;
	  do{n = ugen(0, num_documents);}while(bows[n].size()==0);
	  perp_sample[j] = bows[n];
	}
	std::cout << "approximated perplexity: " << lda.perplexity(perp_sample, max_perp_iter, perp_thresh) << std::endl;
      }
    }
#endif
  }

  show_topics(lda, vocmap, 50);

  // perplexity
  std::vector<bow_type> perp_sample(perp_sample_size); 
  for(size_t j=0;j<perp_sample_size;++j){
    size_t n;
    do{n = ugen(0, num_documents);}while(bows[n].size()==0);
    perp_sample[j] = bows[n];
  }
  std::cout << "approximated perplexity: " << lda.perplexity(perp_sample, last_max_perp_iter, last_perp_thresh) << std::endl;


  std::cout << "finish." << std::endl;

  return 0;
}
