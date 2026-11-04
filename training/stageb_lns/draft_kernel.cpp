// Approximate root-anchored PHY. No inverse, MMSE solve or counterfactual label.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

struct Draft {
  int U,C,P,M;
  double noise;
  std::vector<int64_t> serving,caps,tbs,offsets,root;
  std::vector<double> queue,weight,urgent,weak,received,hint,projected,xs,ys;
  std::vector<double> root_values,gamma,root_rough;
  std::vector<int> root_mask;
  int64_t candidates=0,ue_updates=0,prg_updates=0,roots=0;
  double root_r=0;
  bool ready=false;
  Draft(int u,int c,int p,double n,const int64_t*s,const int64_t*cap,const int64_t*t,
        const double*q,const double*w,const double*ur,const double*we,
        const double*rc,const double*hi,const double*pr,const int64_t*of,
        const double*x,const double*y):U(u),C(c),P(p),M(1<<c),noise(n),
    serving(s,s+u),caps(cap,cap+u*p),tbs(t,t+u*(p+1)*5),offsets(of,of+29),
    queue(q,q+u),weight(w,w+u),urgent(ur,ur+u),weak(we,we+u),
    received(rc,rc+u*p*c),hint(hi,hi+u*p*4),projected(pr,pr+u*p*c*4),
    xs(x,x+of[28]),ys(y,y+of[28]) {}
  size_t gi(int mode,int u,int p,int mask,int layer) const {
    return (((size_t(mode)*U+u)*P+p)*M+mask)*5+layer;
  }
  double term(int u,double e) const {
    const double target=std::min(queue[u],2400.);
    return e*(weight[u]+.5*urgent[u])-.5*std::max(target-e,0.)*urgent[u]
           +.5*std::min(e,target)*weak[u];
  }
  double bler(int m,double db) const {
    db=std::max(db,-4.75);
    const int begin=offsets[m],end=offsets[m+1];
    if(db<xs[end-1])return 1.;
    if(db>xs[begin])return 0.;
    for(int j=end-1;j>begin;--j)if(db<=xs[j-1])
      return ys[j]+(db-xs[j])/(xs[j-1]-xs[j])*(ys[j-1]-ys[j]);
    return 0.;
  }
};

extern "C" {
void* draft_create(int u,int c,int p,double noise,const int64_t*s,const int64_t*cap,
 const int64_t*t,const double*q,const double*w,const double*ur,const double*we,
 const double*rc,const double*hi,const double*pr,const int64_t*of,const double*x,const double*y){
  if(u<=0||p<=0||c<=0||c>8)return nullptr;
  try{return new Draft(u,c,p,noise,s,cap,t,q,w,ur,we,rc,hi,pr,of,x,y);}catch(...){return nullptr;}
}
void draft_destroy(void*ptr){delete static_cast<Draft*>(ptr);}

int draft_root(void*ptr,const int64_t*action,const double*values){
  auto& d=*static_cast<Draft*>(ptr);
  d.root.assign(action,action+d.C*d.P);
  d.root_values.assign(values,values+d.U*8);
  d.root_mask.assign(d.P,0);
  for(int p=0;p<d.P;++p)for(int c=0;c<d.C;++c)
    if(action[c*d.P+p]>=0)d.root_mask[p]|=1<<c;
  d.gamma.assign(size_t(2)*d.U*d.P*d.M*5,0.);
  for(int u=0;u<d.U;++u)for(int p=0;p<d.P;++p){
    const int own=d.serving[u],old=d.root_mask[p];
    double trace_root=4*d.noise;
    std::array<double,4> proj_root;proj_root.fill(d.noise);
    for(int c=0;c<d.C;++c)if(c!=own&&(old&(1<<c))){
      trace_root+=d.received[(u*d.P+p)*d.C+c];
      for(int l=0;l<4;++l)proj_root[l]+=d.projected[((u*d.P+p)*d.C+c)*4+l];
    }
    for(int mask=0;mask<d.M;++mask){
      double trace=4*d.noise,gt=0,gp=0;
      std::array<double,4> proj;proj.fill(d.noise);
      for(int c=0;c<d.C;++c)if(c!=own&&(mask&(1<<c))){
        trace+=d.received[(u*d.P+p)*d.C+c];
        for(int l=0;l<4;++l)proj[l]+=d.projected[((u*d.P+p)*d.C+c)*4+l];
      }
      for(int l=0;l<4;++l){
        const double h=d.hint[(u*d.P+p)*4+l];
        gt+=h*trace_root/trace;gp+=h*proj_root[l]/proj[l];
        d.gamma[d.gi(0,u,p,mask,l+1)]=gt/(l+1);
        d.gamma[d.gi(1,u,p,mask,l+1)]=gp/(l+1);
      }
    }
  }
  d.root_rough.assign(d.U*2,0.);
  d.root_r=0;
  for(int u=0;u<d.U;++u){
    const int n=int(values[u*8]),layer=std::max(0,int(values[u*8+1]));
    for(int mode=0;mode<2;++mode){
      double total=0;
      for(int p=0;p<d.P;++p)if(action[d.serving[u]*d.P+p]==u)
        total+=d.gamma[d.gi(mode,u,p,d.root_mask[p],layer)];
      d.root_rough[mode*d.U+u]=n?10*std::log10(std::max(total/n,1.e-12)):0.;
    }
    d.root_r+=d.term(u,values[u*8+6]);
  }
  d.ready=true;++d.roots;return 0;
}

// features: [B,U,32]; raw: [B,U,7] = dequeue, Etrace, Espatial,
// risk weight, urgency, weak-service weight, target bytes. mask: [B,U].
int draft_eval(void*ptr,int B,const int64_t*actions,const int64_t*mcs,
               double*scores,float*features,double*raw,uint8_t*mask_out){
  auto& d=*static_cast<Draft*>(ptr);if(!d.ready||B<0)return 1;
  for(int b=0;b<B;++b){
    const auto* a=actions+size_t(b)*d.C*d.P;
    std::vector<int> masks(d.P,0);
    std::vector<uint8_t> resource(d.U,0),affected(d.U,0);
    int changed=0,activity=0;
    for(int c=0;c<d.C;++c)for(int p=0;p<d.P;++p){
      const int i=c*d.P+p,old=d.root[i],now=a[i];
      if(now>=0)masks[p]|=1<<c;
      if(old!=now){++changed;if(old>=0)resource[old]=1;if(now>=0)resource[now]=1;}
      activity+=(old>=0)!=(now>=0);
    }
    for(int u=0;u<d.U;++u){
      affected[u]=resource[u];
      for(int p=0;p<d.P&&!affected[u];++p)
        if(a[d.serving[u]*d.P+p]==u &&
           ((masks[p]^d.root_mask[p])&~(1<<d.serving[u])))affected[u]=1;
    }
    scores[b*2]=scores[b*2+1]=0.;
    for(int u=0;u<d.U;++u){
      auto*f=features+(size_t(b)*d.U+u)*32;
      auto*r=raw+(size_t(b)*d.U+u)*7;
      const auto*old=d.root_values.data()+u*8;
      std::fill(f,f+32,0.f);mask_out[size_t(b)*d.U+u]=affected[u];
      r[0]=old[3];r[1]=r[2]=old[6];r[3]=d.weight[u];r[4]=d.urgent[u];
      r[5]=d.weak[u];r[6]=std::min(d.queue[u],2400.);
      if(!affected[u])continue;
      ++d.ue_updates;
      int n=0,layer=4,touched=0;
      for(int p=0;p<d.P;++p)if(a[d.serving[u]*d.P+p]==u){
        ++n;layer=std::min(layer,int(d.caps[u*d.P+p]));
        touched+=masks[p]!=d.root_mask[p];++d.prg_updates;
      }
      if(!n)layer=0;
      double deq=std::min(d.queue[u],double(d.tbs[(u*(d.P+1)+n)*5+layer]));
      std::array<double,2> db{},bler{},e{};
      for(int mode=0;mode<2;++mode){
        double total=0;
        for(int p=0;p<d.P;++p)if(a[d.serving[u]*d.P+p]==u)
          total+=d.gamma[d.gi(mode,u,p,masks[p],layer)];
        db[mode]=n?10*std::log10(std::max(total/n,1.e-12)):0.;
        if(n&&old[0]>0)db[mode]+=old[4]-d.root_rough[mode*d.U+u];
        bler[mode]=n?d.bler(mcs[u],db[mode]):1.;e[mode]=deq*(1-bler[mode]);
        scores[b*2+mode]+=d.term(u,e[mode])-d.term(u,old[6]);
      }
      r[0]=deq;r[1]=e[0];r[2]=e[1];
      const double old_layer=std::max(old[1],0.),old_db=old[0]>0?old[4]:0.;
      const double v[]={old[0]/d.P,double(n)/d.P,old_layer/4,layer/4.,old[3]/2400.,
        deq/2400.,old_db/30,db[0]/30,db[1]/30,std::max(old[5],0.),bler[0],bler[1],
        old[6]/2400.,e[0]/2400.,e[1]/2400.,d.weight[u]/4,std::log1p(d.queue[u]/2400.),
        mcs[u]/27.,d.urgent[u],d.weak[u],(n-old[0])/d.P,(layer-old_layer)/4,
        (db[0]-old_db)/30,(db[1]-db[0])/30,double(resource[u]),double(!resource[u]),
        double(touched)/d.P,std::abs(db[1]-old_db)/30,
        (e[1]-e[0])/2400.,double(old[0]>0),double(changed)/(d.C*d.P),double(activity)/(d.C*d.P)};
      for(int k=0;k<32;++k)f[k]=float(v[k]);
    }
    ++d.candidates;
  }
  return 0;
}
void draft_stats(void*ptr,int64_t*out){
  auto&d=*static_cast<Draft*>(ptr);out[0]=d.candidates;out[1]=d.ue_updates;
  out[2]=d.prg_updates;out[3]=d.roots;
}
}
