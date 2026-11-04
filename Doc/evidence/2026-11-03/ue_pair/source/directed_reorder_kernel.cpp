// Root-causal weak-UE swap chains and disjoint-cell compositions.
// Only existing gamma-bank entries are read; no candidate BLER or exact J.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" int directed_reorders(int U,int C,int P,int T,const int64_t* root,
 const int64_t* serving,const int64_t* quotas,const int64_t* layers,
 const int64_t* caps,const int64_t* targets,const uint8_t* legal,
 const double* gamma,const double* values,const double* weights,
 int depth,int capacity,int64_t* output,int64_t* stats) {
 try {
  if(C>3||depth<1)return -1;
  for(int j=0;j<6;++j)stats[j]=0;
  const int S=C*P;int count=0;
  std::vector<double> sums(U,0),amount(U,0);std::vector<int> witnesses(U,0);
  for(int c=0;c<C;++c)for(int p=0;p<P;++p){int u=root[c*P+p];if(u>=0){
   sums[u]+=gamma[u*P+p];++stats[1];if(caps[u*P+p]==layers[u])++witnesses[u];}}
  for(int u=0;u<U;++u)amount[u]=std::max(1.,weights[u]*values[u*8+3]);
  std::unordered_set<std::string> seen;seen.emplace(reinterpret_cast<const char*>(root),S*sizeof(int64_t));
  auto add=[&](const std::vector<int64_t>& a){
   std::string key(reinterpret_cast<const char*>(a.data()),S*sizeof(int64_t));
   if(!seen.insert(key).second)return;
   if(count>=capacity)throw 1;
   int cells=0;for(int c=0;c<C;++c){bool changed=false;for(int p=0;p<P;++p)changed|=a[c*P+p]!=root[c*P+p];cells+=changed;}
   stats[5]+=cells>1;std::copy(a.begin(),a.end(),output+count*S);++count;
  };
  std::array<std::vector<std::vector<int64_t>>,3> anchors;
  std::array<int,3> anchor_ue={-1,-1,-1};
  const double penalties[3]={0.,.25,1.};
  for(int target=0;target<T;++target){
   int u=targets[target];if(u<0||u>=U||quotas[u]<=0)return -2;
   int c=serving[u];if(c<0||c>=C)return -2;
   if(anchor_ue[c]<0)anchor_ue[c]=u;
   for(double penalty:penalties){
    std::vector<int64_t>a(root,root+S);auto sum=sums;auto witness=witnesses;
    std::vector<uint8_t>used(P,0);std::vector<int64_t>first,last;
    for(int step=0;step<depth;++step){
     int best_out=-1,best_in=-1,best_d=-1;double best=-1.e300,best_du=0,best_dd=0;
     for(int p=0;p<P;++p)if(!used[p]&&a[c*P+p]==u){
      for(int q=0;q<P;++q){int d=a[c*P+q];
       if(used[q]||d<0||d==u)continue;
       ++stats[0];
       if(!legal[(c*P+q)*U+u]||!legal[(c*P+p)*U+d]||caps[u*P+q]<layers[u]||caps[d*P+p]<layers[d])continue;
       int wu=witness[u]-(caps[u*P+p]==layers[u])+(caps[u*P+q]==layers[u]);
       int wd=witness[d]-(caps[d*P+q]==layers[d])+(caps[d*P+p]==layers[d]);
       if(wu<1||wd<1)continue;
       double du=gamma[u*P+q]-gamma[u*P+p],dd=gamma[d*P+p]-gamma[d*P+q];stats[1]+=4;
       if(du<=1.e-12)continue;
       double gain=du/std::max(sum[u],1.e-12)-penalty*std::max(0.,-dd)/std::max(sum[d],1.e-12)*amount[d]/amount[u];
       if(gain>best){best=gain;best_out=p;best_in=q;best_d=d;best_du=du;best_dd=dd;}
      }
     }
     if(best_out<0)break;
     int p=best_out,q=best_in,d=best_d;
     witness[u]+=(caps[u*P+q]==layers[u])-(caps[u*P+p]==layers[u]);
     witness[d]+=(caps[d*P+p]==layers[d])-(caps[d*P+q]==layers[d]);
     std::swap(a[c*P+p],a[c*P+q]);used[p]=used[q]=1;sum[u]+=best_du;sum[d]+=best_dd;
     ++stats[2];add(a);if(first.empty())first=a;last=a;
    }
    if(anchor_ue[c]==u&&!first.empty()){
     anchors[c].push_back(first);if(first!=last)anchors[c].push_back(last);
    }
   }
  }
  // Each component changes a disjoint serving cell. Counts, exact layers and
  // activity remain valid when combining them; no cross-cell UE reassignment.
  for(int c=0;c<C;++c)for(int d=c+1;d<C;++d)
   for(const auto&a:anchors[c])for(const auto&b:anchors[d]){
    std::vector<int64_t>joint=a;std::copy(b.begin()+d*P,b.begin()+(d+1)*P,joint.begin()+d*P);
    ++stats[3];add(joint);
   }
  if(C==3&&!anchors[0].empty()&&!anchors[1].empty()&&!anchors[2].empty())
   for(int mask=0;mask<8;++mask){
    std::vector<int64_t>joint(root,root+S);
    for(int c=0;c<C;++c){const auto&a=(mask&(1<<c))?anchors[c].back():anchors[c].front();
     std::copy(a.begin()+c*P,a.begin()+(c+1)*P,joint.begin()+c*P);}
    ++stats[3];add(joint);
   }
  stats[4]=count;return count;
 }catch(...){return -3;}
}
