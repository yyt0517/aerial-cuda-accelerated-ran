// Cheap one-PRG DPLS candidate features. No candidate MMSE or BLER lookup.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int F = 48;
constexpr int V = 8;
struct Context {
    int U,C,P;
    double noise;
    std::vector<int64_t> serving,mcs,caps,tbs,root;
    std::vector<uint8_t> legal;
    std::vector<double> queue,weights,received,quality,log_queue,values,denominator,reliability,failure;
    std::vector<int> count,layer,minimum,multiplicity,second;
    double root_j=0.,root_d=0.,root_activity=0.;
    bool ready=false;
    Context(int u,int c,int p,double n,const int64_t* s,const int64_t* m,
            const int64_t* cap,const int64_t* tb,const uint8_t* le,
            const double* q,const double* w,const double* rec,const double* qual):
        U(u),C(c),P(p),noise(n),serving(s,s+u),mcs(m,m+u),caps(cap,cap+u*p),
        tbs(tb,tb+u*(p+1)*5),root(c*p),legal(le,le+c*p*u),queue(q,q+u),weights(w,w+u),
        received(rec,rec+u*p*c),quality(qual,qual+u*p),log_queue(u),values(u*V),denominator(u*p),
        reliability(u),failure(u),count(u),layer(u),minimum(u),multiplicity(u),second(u) {
        for(int i=0;i<u;++i)log_queue[i]=std::log1p(queue[i]/3000.);
        for(double& value:quality)value=std::log1p(value);
    }

    int set_base(const int64_t* a,const double* v) {
        root.assign(a,a+C*P);values.assign(v,v+U*V);
        root_j=root_d=root_activity=0.;
        for(int i=0;i<C*P;++i)root_activity+=root[i]>=0;
        root_activity/=double(C*P);
        for(int u=0;u<U;++u){
            int n=0,lo=5,lo_count=0,hi=5;
            for(int p=0;p<P;++p)if(root[serving[u]*P+p]==u){
                ++n;int cap=int(caps[u*P+p]);
                if(cap<lo){hi=lo;lo=cap;lo_count=1;}
                else if(cap==lo)++lo_count;
                else if(cap<hi)hi=cap;
            }
            count[u]=n;layer[u]=n?int(std::max(v[u*V+1],0.)):4;
            minimum[u]=lo;multiplicity[u]=lo_count;second[u]=hi;
            failure[u]=n?std::max(v[u*V+5],0.):.5;
            reliability[u]=n?1-failure[u]:.5;
            root_j+=v[u*V+6]*weights[u];root_d+=v[u*V+3];
            for(int p=0;p<P;++p){
                double d=4*noise;
                for(int c=0;c<C;++c)if(root[c*P+p]>=0)d+=received[(u*P+p)*C+c];
                denominator[u*P+p]=d;
            }
        }
        ready=true;return 0;
    }

    int evaluate(int N,const int64_t* actions,double floor,float* output,double* heuristic,double* totals){
        if(!ready||N<0||!std::isfinite(floor)||floor<0)return 1;
        std::vector<int> dc(U),da(C);
        for(int b=0;b<N;++b){
            const int64_t* a=actions+b*C*P;
            int prg=-1,edits=0,muted=0,activated=0,swaps=0;
            std::fill(dc.begin(),dc.end(),0);std::fill(da.begin(),da.end(),0);
            for(int c=0;c<C;++c)for(int p=0;p<P;++p){
                int64_t old=root[c*P+p],value=a[c*P+p];
                if(value==old)continue;
                if(value < -1 || value >= U)return 2;
                if(value>=0&&!legal[(c*P+p)*U+value])return 2;
                if(prg>=0&&prg!=p)return 3;
                prg=p;if(++edits>2)return 3;
                if(old>=0)--dc[old];if(value>=0)++dc[value];
                da[c]=int(value>=0)-int(old>=0);
                muted+=da[c]<0;activated+=da[c]>0;swaps+=old>=0&&value>=0;
            }
            if(prg<0)prg=0;
            std::array<double,F>x{};
            double fixed_total=0.,activity_total=0.,total_d=0.,donors=0.,recipients=0.,victims=0.;
            for(int u=0;u<U;++u){
                bool donor=dc[u]<0,recipient=dc[u]>0;
                int n=count[u]+dc[u],cap=int(caps[u*P+prg]);
                bool unique_min=cap==minimum[u]&&multiplicity[u]==1;
                int l=layer[u];
                if(donor&&unique_min)l=second[u];
                if(recipient)l=std::min(l,cap);
                if(n==0)l=0;
                if(n<0||n>P||l<0||l>4)return 4;
                double dequeue=std::min(queue[u],double(tbs[(u*(P+1)+n)*5+l]));
                double dd=dequeue-values[u*V+3],fixed=dd*reliability[u]*weights[u];
                total_d+=dequeue;fixed_total+=fixed;
                bool selected=a[serving[u]*P+prg]==u;
                double tb_ratio=0.,activity=0.;bool victim=false;
                if(selected&&(muted||activated)){
                    double before=denominator[u*P+prg],after=before;
                    for(int c=0;c<C;++c)after+=da[c]*received[(u*P+prg)*C+c];
                    double ratio=std::log(before/std::max(after,1.e-30));
                    tb_ratio=ratio/std::max(n,1);victim=std::abs(ratio)>1.e-12;
                    double risk=std::min(queue[u],std::max(dequeue,values[u*V+3]))*failure[u]*weights[u];
                    activity=.5*risk*std::tanh(tb_ratio);
                }
                activity_total+=activity;
                int old_l=count[u]?layer[u]:0,dl=l-old_l;
                x[3]+=dd;x[4]+=std::max(dd,0.);x[5]+=std::min(dd,0.);
                x[6]+=dc[u]!=0;x[17]+=std::max(dc[u],0);x[18]+=std::min(dc[u],0);
                x[19]+=std::max(dl,0);x[20]+=std::min(dl,0);x[21]+=donor&&unique_min;
                x[22]+=count[u]==0&&n>0;x[23]+=count[u]>0&&n==0;
                x[24]+=values[u*V+3]<queue[u]&&dequeue>=queue[u];
                x[25]+=std::max(activity,0.);x[26]+=std::min(activity,0.);
                if(victim){++victims;x[27]+=tb_ratio;}
                x[28]=std::max(x[28],tb_ratio);x[29]=std::min(x[29],tb_ratio);
                if(recipient){
                    ++recipients;x[30]+=count[u];x[32]+=old_l;x[34]+=mcs[u];x[36]+=log_queue[u];
                    x[38]+=quality[u*P+prg];x[40]+=cap;x[42]+=failure[u];x[44]+=weights[u];
                }
                if(donor){
                    ++donors;x[31]+=count[u];x[33]+=old_l;x[35]+=mcs[u];x[37]+=log_queue[u];
                    x[39]+=quality[u*P+prg];x[41]+=cap;x[43]+=failure[u];x[45]+=weights[u];
                }
                x[46]=std::max(x[46],fixed);x[47]=std::min(x[47],fixed);
            }
            heuristic[b]=fixed_total+activity_total;totals[b]=total_d;
            x[0]=heuristic[b]/3000.;x[1]=fixed_total/3000.;x[2]=activity_total/3000.;
            for(int i:{3,4,5,25,26,46,47})x[i]/=3000.;
            x[6]/=U;x[7]=double(edits)/(C*P);x[8]=double(muted)/C;x[9]=double(activated)/C;
            x[10]=double(swaps)/C;x[11]=double(muted+activated)/C;x[12]=victims/U;
            x[13]=root_j/3000.;x[14]=(root_d-floor)/3000.;x[15]=root_activity;x[16]=(total_d-floor)/3000.;
            x[17]/=P;x[18]/=P;x[19]/=4.;x[20]/=4.;
            for(int i:{21,22,23,24})x[i]/=U;
            x[27]/=std::max(victims,1.)*3.;x[28]/=3.;x[29]/=3.;
            for(int i:{30,32,34,36,38,40,42,44})x[i]/=std::max(recipients,1.);
            for(int i:{31,33,35,37,39,41,43,45})x[i]/=std::max(donors,1.);
            x[30]/=P;x[31]/=P;
            for(int i:{32,33,40,41,44,45})x[i]/=4.;
            x[34]/=27.;x[35]/=27.;
            for(int i=0;i<F;++i){if(!std::isfinite(x[i]))return 5;output[b*F+i]=float(x[i]);}
            if(!std::isfinite(heuristic[b])||!std::isfinite(total_d))return 5;
        }
        return 0;
    }
};
}

extern "C" {
void* edit_features_create(int u,int c,int p,double noise,const int64_t* serving,const int64_t* mcs,
        const int64_t* caps,const int64_t* tbs,const uint8_t* legal,const double* queue,const double* weights,
        const double* received,const double* quality){
    try{return new Context(u,c,p,noise,serving,mcs,caps,tbs,legal,queue,weights,received,quality);}
    catch(...){return nullptr;}
}
void edit_features_destroy(void* context){delete static_cast<Context*>(context);}
int edit_features_base(void* context,const int64_t* action,const double* values){
    if(!context)return 1;return static_cast<Context*>(context)->set_base(action,values);
}
int edit_features_evaluate(void* context,int n,const int64_t* actions,double floor,
        float* features,double* heuristic,double* dequeue){
    if(!context)return 1;return static_cast<Context*>(context)->evaluate(n,actions,floor,features,heuristic,dequeue);
}
}
