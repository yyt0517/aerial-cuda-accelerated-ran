// Joint-condition reorder enumeration and sparse gamma-delta proxy.
// No candidate native BLER or exact complete-schedule J is evaluated here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" int joint_reorders(int U,int C,int P,const int64_t* root,const int64_t* quotas,
 const int64_t* layers,const int64_t* caps,const uint8_t* legal,const uint8_t* activity,
 int wide,const int64_t* orders,int K,int capacity,int64_t* output,int64_t* stats) {
 try {
  int S=C*P,count=0;stats[0]=stats[1]=0;std::unordered_set<std::string> seen;
  seen.emplace(reinterpret_cast<const char*>(root),S*sizeof(int64_t));
  std::vector<int> witnesses(U,0);
  for(int c=0;c<C;++c)for(int p=0;p<P;++p){int u=root[c*P+p];if(u>=0&&caps[u*P+p]==layers[u])++witnesses[u];}
  auto add=[&](int cell,const int* pos,int length,int shift){
   ++stats[0];std::vector<int64_t> a(root,root+S);std::vector<int> change(U,0);
   for(int j=0;j<length;++j)a[cell*P+pos[j]]=root[cell*P+pos[(j-shift+length)%length]];
   for(int j=0;j<length;++j){int p=pos[j],old=root[cell*P+p],now=a[cell*P+p];if(old==now)continue;
    if(now>=0&&(!legal[(cell*P+p)*U+now]||caps[now*P+p]<layers[now]))return;
    if(old>=0&&caps[old*P+p]==layers[old])--change[old];
    if(now>=0&&caps[now*P+p]==layers[now])++change[now];
   }
   for(int u=0;u<U;++u)if(quotas[u]>0&&witnesses[u]+change[u]<1)return;
   std::string key(reinterpret_cast<const char*>(a.data()),S*sizeof(int64_t));if(seen.count(key))return;
   seen.insert(key);if(count>=capacity)throw 1;std::copy(a.begin(),a.end(),output+count*S);++count;
  };
  for(int c=0;c<C;++c){
   std::vector<int> prgs;for(int p=0;p<P;++p)if(activity[c*P+p])prgs.push_back(p);int M=prgs.size();
   for(int i=0;i<M;++i)for(int j=i+1;j<M;++j){int pos[2]={prgs[i],prgs[j]};add(c,pos,2,1);}
   int T=wide?M:std::min(M,6);
   for(int i=0;i<T;++i)for(int j=i+1;j<T;++j)for(int k=j+1;k<T;++k){
    int pos[3]={prgs[i],prgs[j],prgs[k]};add(c,pos,3,1);add(c,pos,3,2);
   }
   if(wide&&M>=4)for(int order=0;order<K;++order)for(int start=0;start<M;++start){
    int pos[4];for(int j=0;j<4;++j)pos[j]=int(orders[(c*K+order)*P+(start+j)%M]);
    for(int shift=1;shift<4;++shift)add(c,pos,4,shift);
   }
  }
  stats[1]=count;return count;
 } catch(...){return -1;}
}

extern "C" int joint_delta(int N,int U,int C,int P,const int64_t* actions,const int64_t* root,
 const int64_t* quotas,const int64_t* layers,const int64_t* caps,const uint8_t* legal,
 const double* gamma,const double* threshold,const double* weights,const double* values,
 double* score,double* projected,uint8_t* valid,int64_t* stats) {
 try {
  auto sigmoid=[](double x){return 1./(1.+std::exp(-std::max(-30.,std::min(30.,x))));};
  std::vector<double> sum(U,0),correction(U,0);std::vector<int> witnesses(U,0);stats[0]=0;
  for(int c=0;c<C;++c)for(int p=0;p<P;++p){int u=root[c*P+p];if(u>=0){sum[u]+=gamma[u*P+p];++stats[0];if(caps[u*P+p]==layers[u])++witnesses[u];}}
  double root_j=0;for(int u=0;u<U;++u){root_j+=weights[u]*values[u*8+6];
   correction[u]=quotas[u]>0?(1-std::max(0.,values[u*8+5]))-sigmoid((values[u*8+4]-threshold[u])/1.5):0.;}
  for(int n=0;n<N;++n){
   const auto* a=actions+n*C*P;auto* e=projected+n*U;std::vector<double> delta(U,0);
   std::vector<int> changed(U,0),dn(U,0),dw(U,0);bool ok=true;
   for(int u=0;u<U;++u)e[u]=values[u*8+6];
   for(int c=0;c<C;++c)for(int p=0;p<P;++p){int old=root[c*P+p],now=a[c*P+p];if(old==now)continue;
    if((old>=0)!=(now>=0)){ok=false;continue;}
    if(now>=0&&(!legal[(c*P+p)*U+now]||caps[now*P+p]<layers[now])){ok=false;continue;}
    if(old>=0){changed[old]=1;--dn[old];delta[old]-=gamma[old*P+p];++stats[0];if(caps[old*P+p]==layers[old])--dw[old];}
    if(now>=0){changed[now]=1;++dn[now];delta[now]+=gamma[now*P+p];++stats[0];if(caps[now*P+p]==layers[now])++dw[now];}
   }
   for(int u=0;u<U;++u)if(dn[u]||(quotas[u]>0&&witnesses[u]+dw[u]<1))ok=false;
   valid[n]=ok;score[n]=root_j;if(!ok)continue;
   for(int u=0;u<U;++u)if(changed[u]){
    double mean=(sum[u]+delta[u])/std::max<int64_t>(quotas[u],1);
    double db=10*std::log10(std::max(mean,1.e-30));
    double reliability=std::max(0.,std::min(1.,sigmoid((db-threshold[u])/1.5)+correction[u]));
    e[u]=values[u*8+3]*reliability;score[n]+=weights[u]*(e[u]-values[u*8+6]);
   }
  }
  return 0;
 } catch(...){return -1;}
}
