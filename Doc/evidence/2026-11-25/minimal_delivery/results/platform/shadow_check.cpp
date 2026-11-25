#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
int main(){
 std::cout<<"seed,case,n,mean_db,sample_std_db,theoretical_std_db\n";
 for(int seed=41;seed<=50;seed++) for(int fix=0;fix<2;fix++){
  std::mt19937 gen(seed);float sigma=fix?3.0f*std::log(10.0f)/10.0f:3.0f;
  std::lognormal_distribution<float> dist(0.0f,sigma);double sum=0,squares=0;int n=1000000;
  for(int i=0;i<n;i++){double v=10.0*std::log10(dist(gen));sum+=v;squares+=v*v;}
  std::cout<<seed<<","<<(fix?"corrected":"legacy")<<","<<n<<","<<std::setprecision(12)<<sum/n<<","<<std::sqrt((squares-sum*sum/n)/(n-1))<<","<<sigma*10/std::log(10.)<<"\n";
 }
}
