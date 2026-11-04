// Cheap sequential TB bookkeeping after one joint neural score calculation.
// No candidate MMSE, BLER, neural forward pass, or acceptance takes place here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>

extern "C" int masked_decode(int U, int C, int P, int L, int K, double noise,
 const int64_t* root, const int64_t* block, const int64_t* serving,
 const int64_t* caps, const int64_t* tbs, const double* queue,
 const double* received, const uint8_t* legal, const float* logits,
 const float* beta, const int64_t* modes, const double* random, int64_t* output) {
 try {
  const int N=C*P, A=U+1;
  for(int k=0;k<K;++k) {
   int64_t* a=output+k*N;
   std::copy(root,root+N,a);
   for(int j=0;j<L;++j) a[block[j]]=-1;
   std::vector<int> count(U,0), layer(U,4);
   for(int i=0;i<N;++i) if(a[i]>=0) {
    int u=a[i]; ++count[u]; layer[u]=std::min(layer[u],int(caps[u*P+i%P]));
   }
   int h=modes[k];
   for(int j=0;j<L;++j) {
    int slot=block[j], p=slot%P, best=0;
    double best_score=-std::numeric_limits<double>::infinity();
    for(int v=0;v<A;++v) if(legal[slot*A+v]) {
     float phi[10]={0};
     phi[8]=v==0; phi[9]=v==root[slot]+1;
     if(v>0) {
      int u=v-1, n=std::min(count[u]+1,P), l=std::min(layer[u],int(caps[u*P+p]));
      double before=std::min(queue[u],double(tbs[(u*(P+1)+count[u])*5+layer[u]]));
      double after=std::min(queue[u],double(tbs[(u*(P+1)+n)*5+l]));
      double interference=4*noise;
      for(int c=0;c<C;++c) if(c!=serving[u] && a[c*P+p]>=0) interference+=received[(u*P+p)*C+c];
      double signal=received[(u*P+p)*C+serving[u]];
      phi[0]=double(count[u])/P; phi[1]=double(n)/P;
      phi[2]=double(layer[u])/4; phi[3]=double(l)/4;
      phi[4]=before/std::max(queue[u],3000.); phi[5]=after/std::max(queue[u],3000.);
      phi[6]=(after-before)/3000.; phi[7]=std::log1p(signal/std::max(interference,1.e-30))/8;
     }
     float correction=0;
     for(int f=0;f<10;++f) correction+=phi[f]*beta[h*10+f];
     double score=double(logits[(h*L+j)*A+v]+correction)+random[(k*L+j)*A+v];
     if(score>best_score) {best_score=score;best=v;}
    }
    a[slot]=best-1;
    if(best>0) {int u=best-1;++count[u];layer[u]=std::min(layer[u],int(caps[u*P+p]));}
   }
  }
  return 0;
 } catch(...) { return -1; }
}
