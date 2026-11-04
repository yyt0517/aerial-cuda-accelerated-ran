// Logical TB descriptors only. No MMSE, candidate SINR/BLER or PHY context.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" int selective_tb_features(int N,int U,int C,int P,
 const int64_t* actions,const int64_t* root,const int64_t* serving,
 const int64_t* caps,const int64_t* histogram,const int64_t* tbs,
 const double* queue,const double* weights,const double* values,
 const double* received,const double* denominator,double floor,
 int64_t* counts,int64_t* layers,double* dequeue,double* proxy,double* total,float* x) {
 try {
  if(U<1||C<1||P<1||N<0)return 1;
  double root_j=0,root_d=0;
  for(int u=0;u<U;++u){root_j+=weights[u]*values[u*8+6];root_d+=values[u*8+3];}
  std::vector<int> positive;
  for(int u=0;u<U;++u)if(values[u*8+6]>1.)positive.push_back(u);
  std::stable_sort(positive.begin(),positive.end(),[&](int a,int b){return values[a*8+6]<values[b*8+6];});
  std::vector<int> weak(U,0);double weak_sum=0;
  int weak_count=positive.empty()?0:std::max(1,int(std::ceil(.2*positive.size())));
  for(int j=0;j<weak_count;++j){weak[positive[j]]=1;weak_sum+=values[positive[j]*8+6];}
  for(int n=0;n<N;++n){
   const auto* a=actions+n*C*P;auto* co=counts+n*U;auto* la=layers+n*U;
   auto* de=dequeue+n*U;auto* f=x+n*60;std::fill(f,f+60,0.f);
   std::vector<int64_t> hist(histogram,histogram+U*5);
   std::vector<int> affected,changed,activity;
   int edits=0,activity_edits=0;
   auto add=[&](int u){if(u>=0&&std::find(affected.begin(),affected.end(),u)==affected.end())affected.push_back(u);};
   for(int u=0;u<U;++u)co[u]=int64_t(values[u*8]);
   for(int c=0;c<C;++c)for(int p=0;p<P;++p){
    int old=root[c*P+p],now=a[c*P+p];if(old==now)continue;
    ++edits;add(old);add(now);
    if(std::find(changed.begin(),changed.end(),p)==changed.end())changed.push_back(p);
    if((old>=0)!=(now>=0)){
     ++activity_edits;if(std::find(activity.begin(),activity.end(),p)==activity.end())activity.push_back(p);
    }
    if(old>=0){--co[old];--hist[old*5+caps[old*P+p]];}
    if(now>=0){++co[now];++hist[now*5+caps[now*P+p]];}
   }
   if(edits>3||affected.size()>6)return 2;
   double fixed=0,interference=0,tot=0,projected_weak=0,min_service_ratio=1.;int lost=0,gained=0;
   for(int u=0;u<U;++u){
    la[u]=0;if(co[u])for(int l=1;l<=4;++l)if(hist[u*5+l]){la[u]=l;break;}
    if(co[u]<0||co[u]>P||la[u]<0||la[u]>4)return 3;
    de[u]=std::min(queue[u],double(tbs[(u*(P+1)+co[u])*5+la[u]]));tot+=de[u];
    double reliability=values[u*8]>0?1-std::max(0.,values[u*8+5]):.5;
    fixed+=weights[u]*(de[u]*reliability-values[u*8+6]);
    if(weak[u])projected_weak+=de[u]*reliability;
    if(values[u*8+6]>1.)min_service_ratio=std::min(min_service_ratio,de[u]*reliability/values[u*8+6]);
    lost+=values[u*8]>0&&co[u]==0;gained+=values[u*8]==0&&co[u]>0;
    double logsum=0;
    for(int p:activity)if(a[serving[u]*P+p]==u){
     double after=denominator[u*P+p];
     for(int c=0;c<C;++c){int delta=int(a[c*P+p]>=0)-int(root[c*P+p]>=0);after+=delta*received[(u*P+p)*C+c];}
     logsum+=std::log(denominator[u*P+p]/std::max(after,1.e-30));
    }
    double risk=std::max(de[u],values[u*8+3])*(1-reliability);
    interference+=.5*weights[u]*risk*std::tanh(logsum/std::max<int64_t>(co[u],1));
   }
   proxy[n]=fixed+interference;total[n]=tot;
   f[0]=edits/3.f;f[1]=changed.size()/3.f;f[2]=activity_edits/3.f;
   f[3]=(tot-root_d)/3000.;f[4]=fixed/3000.;f[5]=interference/3000.;
   f[6]=(tot-floor)/3000.;f[7]=double(lost)/U;f[8]=double(gained)/U;
   f[9]=weak_sum>0?(projected_weak-weak_sum)/weak_sum:0.;f[10]=min_service_ratio;
   // f9/f10 project current-root service with frozen reliability; not labels.
   // f11: candidate family, assigned by caller
   for(size_t j=0;j<affected.size();++j){int u=affected[j];auto* s=f+12+j*8;
    s[0]=values[u*8]/P;s[1]=double(co[u])/P;
    s[2]=std::max(0.,values[u*8+1])/4.;s[3]=double(la[u])/4;
    s[4]=values[u*8+3]/3000.;s[5]=de[u]/3000.;
    s[6]=values[u*8]>0?1-std::max(0.,values[u*8+5]):.5;s[7]=weights[u]/4.;
   }
  }
  return 0;
 } catch(...){return 9;}
}
