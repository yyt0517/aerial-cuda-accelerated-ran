// Separate diagnostic extension. The canonical physics source is unchanged.
#include "physics_kernel.cpp"

extern "C" {
int rc_bank(void* ptr,double* output) {
 try {
  auto& c=*static_cast<Context*>(ptr);if(c.C>3)return 1;
  int M=1<<c.C;
  for(int u=0;u<c.U;++u)for(int p=0;p<c.P;++p)for(int m=0;m<M;++m){
   const auto& g=c.gamma(u,p,m);double sum=0;
   for(int l=0;l<4;++l){sum+=g[l];output[((u*c.P+p)*M+m)*4+l]=sum/(l+1);}
  }
  return 0;
 } catch(...){return 9;}
}

// Approximate ranking only: root-anchored smooth reliability, never native
// candidate BLER or complete exact J. All gamma cache requests are counted.
int rc_proxy(void* ptr,int N,const int64_t* actions,const double* root_values,
             const double* weights,double* score,double* total,int64_t* counts,int64_t* layers) {
 try {
  auto& c=*static_cast<Context*>(ptr);if(c.C>3)return 1;
  std::vector<double> threshold(c.U),correction(c.U);
  auto sigmoid=[](double x){return 1./(1.+std::exp(-std::max(-30.,std::min(30.,x))));};
  for(int u=0;u<c.U;++u){
   int begin=c.offsets[c.mcs[u]],end=c.offsets[c.mcs[u]+1],closest=begin;
   for(int k=begin+1;k<end;++k)if(std::abs(c.bler[k]-.5)<std::abs(c.bler[closest]-.5))closest=k;
   threshold[u]=c.snr[closest];
   correction[u]=root_values[u*FIELDS+COUNT]>0?
    (1-std::max(0.,root_values[u*FIELDS+BLER]))-sigmoid((root_values[u*FIELDS+SINR]-threshold[u])/1.5):0.;
  }
  for(int n=0;n<N;++n){
   const auto* a=actions+n*c.C*c.P;std::vector<uint64_t> masks(c.P,0);
   for(int p=0;p<c.P;++p)for(int cell=0;cell<c.C;++cell)if(a[cell*c.P+p]>=0)masks[p]|=uint64_t(1)<<cell;
   score[n]=total[n]=0;
   for(int u=0;u<c.U;++u){
    int count=0,layer=4;
    for(int p=0;p<c.P;++p)if(a[c.serving[u]*c.P+p]==u){++count;layer=std::min(layer,int(c.caps[u*c.P+p]));}
    counts[n*c.U+u]=count;layers[n*c.U+u]=count?layer:0;if(!count)continue;
    double sum=0;
    for(int p=0;p<c.P;++p)if(a[c.serving[u]*c.P+p]==u){const auto& g=c.gamma(u,p,masks[p]);for(int l=0;l<layer;++l)sum+=g[l];}
    double db=10*std::log10(std::max(sum/(count*layer),1.e-30));
    double d=std::min(c.queue[u],double(c.tbs[(u*(c.P+1)+count)*5+layer]));
    double reliability=std::max(0.,std::min(1.,sigmoid((db-threshold[u])/1.5)+correction[u]));
    total[n]+=d;score[n]+=weights[u]*d*reliability;
   }
  }
  return 0;
 } catch(...){return 9;}
}
}
