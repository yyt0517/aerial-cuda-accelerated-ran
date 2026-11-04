// Equivalent ordered candidate enumeration; no PHY or candidate scoring.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" int resource_pool(int U,int C,int P,const int64_t* root,const double* values,
 const double* weights,const uint8_t* legal,const double* static_utility,
 int quotas,int fixed_activity,const uint8_t* activity,int capacity,int64_t* output) {
 try {
  const int S=C*P;int count=0;std::unordered_set<std::string> seen;
  seen.emplace(reinterpret_cast<const char*>(root),S*sizeof(int64_t));
  auto add=[&](const std::vector<int64_t>& a){
   std::string key(reinterpret_cast<const char*>(a.data()),S*sizeof(int64_t));if(seen.count(key))return;
   for(int c=0;c<C;++c)for(int p=0;p<P;++p){int u=a[c*P+p];if(u>=0&&!legal[(c*P+p)*U+u])return;}
   seen.insert(key);if(count>=capacity)throw 1;
   std::copy(a.begin(),a.end(),output+count*S);++count;
  };
  if(!quotas)for(int c=0;c<C;++c)for(int p=0;p<P;++p){
   if(fixed_activity&&!activity[c*P+p])continue;
   std::vector<std::pair<double,int>> options;
   for(int u=0;u<U;++u)if(legal[(c*P+p)*U+u]){
    // Input supplies dynamic utility to retain NumPy arithmetic/tie identity.
    options.emplace_back(static_utility[u*P+p],u);
   }
   std::stable_sort(options.begin(),options.end(),[](const auto& a,const auto& b){return a.first>b.first;});
   std::vector<int> choices{int(root[c*P+p]),-1};
   for(int j=0;j<std::min(3,int(options.size()));++j)choices.push_back(options[j].second);
   std::vector<int> used;
   for(int u:choices){
    if(std::find(used.begin(),used.end(),u)!=used.end())continue;used.push_back(u);
    if(u<0&&fixed_activity)continue;
    std::vector<int64_t> a(root,root+S);a[c*P+p]=u;add(a);
   }
  }
  for(int c=0;c<C;++c){
   std::vector<int> prgs;for(int p=0;p<P;++p)if(!fixed_activity||activity[c*P+p])prgs.push_back(p);
   for(size_t j=0;j<prgs.size();++j)for(size_t k=j+1;k<prgs.size();++k){
    int p=prgs[j],q=prgs[k];if(root[c*P+p]==root[c*P+q])continue;
    std::vector<int64_t> a(root,root+S);std::swap(a[c*P+p],a[c*P+q]);add(a);
   }
   int limit=std::min(6,int(prgs.size()));
   for(int j=0;j<limit;++j)for(int k=j+1;k<limit;++k)for(int l=k+1;l<limit;++l){
    int pos[3]={prgs[j],prgs[k],prgs[l]};
    for(int shift=1;shift<=2;++shift){
     std::vector<int64_t> a(root,root+S);
     for(int t=0;t<3;++t)a[c*P+pos[t]]=root[c*P+pos[(t-shift+3)%3]];
     add(a);
    }
   }
  }
  return count;
 } catch(...){return -1;}
}
