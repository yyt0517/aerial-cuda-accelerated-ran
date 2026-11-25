// Persistent one-PRG edit table, sparse exact PHY payloads and integer J deltas.
// This translation unit reuses the unchanged native physical model.
#include "physics_kernel.cpp"
#include <chrono>
#include <cfenv>
#include <stdexcept>
#include <string>

namespace pd {
using Clock=std::chrono::steady_clock;
constexpr int64_t SCALE=1000000;
thread_local std::string error;
using Observer=int(*)(void*,const int64_t*,const int64_t*,const double*,const int64_t*,
                      const int64_t*,const int64_t*,const int64_t*,const double*,const int64_t*);

struct Edit {
    int prg;
    int64_t key=0,dj=0,dd=0;
    __uint128_t support=0,footprint=0;
    bool valid=false;
    std::vector<int64_t> target;
    std::vector<int> users;
    std::vector<std::array<double,FIELDS>> payload;
};

struct Solver {
    Context& phy;
    const uint8_t* legal;
    const double* weights;
    std::vector<int64_t> action,contribution;
    std::vector<double> values;
    std::vector<uint64_t> masks;
    std::vector<std::vector<Edit>> table;
    int64_t score=0,dequeue=0,floor=0;
    __uint128_t weak=0;
    int mode; //0 full-refresh;1 exact cache;2 unguarded cap4;3 guarded cap4
    bool profile;
    int64_t stats[18]{};
    double times[8]{};

    int64_t quant(int u,double expected) const {
        double x=(expected*weights[u])*double(SCALE);
        if(!std::isfinite(x)||x<0||x>double(INT64_MAX/4))throw std::overflow_error("Unsafe contribution");
        return int64_t(std::nearbyint(x)); // explicitly FE_TONEAREST, ties to even
    }
    int64_t bytes(double x) const {
        if(x<0||x!=std::floor(x)||x>double(INT64_MAX/100))throw std::runtime_error("Nonintegral/unsafe dequeue");
        return int64_t(x);
    }
    void totals(){
        score=dequeue=0;
        for(int u=0;u<phy.U;++u){contribution[u]=quant(u,values[u*FIELDS+EXPECTED]);score+=contribution[u];dequeue+=bytes(values[u*FIELDS+DEQUEUE]);}
    }
    void mask(int p){
        masks[p]=0;
        for(int c=0;c<phy.C;++c)if(action[c*phy.P+p]>=0)masks[p]|=uint64_t(1)<<c;
    }
    Solver(Context& p,const uint8_t* l,const double* w,const int64_t* raw,const int64_t* warm,
           int m,uint64_t weakmask,bool prof):phy(p),legal(l),weights(w),
        action(raw,raw+p.C*p.P),contribution(p.U),values(p.U*FIELDS),masks(p.P),table(p.P),weak(weakmask),mode(m),profile(prof){
        if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("Requires nearest-even rounding environment");
        long double bound=0;
        for(int u=0;u<p.U;++u){
            if(!std::isfinite(w[u])||w[u]<=0)throw std::runtime_error("Invalid weight");
            bytes(p.queue[u]);bound+=std::ceil((long double)p.queue[u]*w[u]*SCALE)+1;
        }
        if(bound>=(long double)INT64_MAX/4)throw std::overflow_error("Quantized objective bound too large");
        phy.evaluate(action.data(),nullptr,nullptr,values.data());stats[9]=1;totals();
        floor=(90*dequeue+99)/100;
        if(warm){
            std::vector<double> v(p.U*FIELDS);phy.evaluate(warm,nullptr,nullptr,v.data());++stats[9];
            int64_t j=0,d=0;for(int u=0;u<p.U;++u){j+=quant(u,v[u*FIELDS+EXPECTED]);d+=bytes(v[u*FIELDS+DEQUEUE]);}
            if(d>=floor&&j>score){action.assign(warm,warm+p.C*p.P);values.swap(v);totals();}
        }
        for(int q=0;q<p.P;++q)mask(q);
    }
    void add(int p,int source,int receiver,int target){
        Edit e;e.prg=p;e.target.assign(phy.C,-2);
        if(source>=0)e.target[source]=-1;
        if(receiver>=0)e.target[receiver]=target;
        bool activity=false;
        for(int c=0;c<phy.C;++c){
            auto old=action[c*phy.P+p],next=e.target[c]==-2?old:e.target[c];
            if(old!=next){if(old>=0)e.support|=__uint128_t(1)<<old;if(next>=0)e.support|=__uint128_t(1)<<next;activity|=(old>=0)!=(next>=0);}
        }
        if(activity)for(int c=0;c<phy.C;++c){
            auto old=action[c*phy.P+p],next=e.target[c]==-2?old:e.target[c];
            if(old>=0)e.support|=__uint128_t(1)<<old;if(next>=0)e.support|=__uint128_t(1)<<next;
        }
        e.key=p;for(auto t:e.target)e.key=e.key*(phy.U+3)+t+2;
        e.footprint=e.support|(__uint128_t(1)<<(phy.U+p));
        for(int u=0;u<phy.U;++u)if((e.support>>u)&1)e.users.push_back(u);
        e.payload.resize(e.users.size());table[p].push_back(std::move(e));++stats[2];
    }
    void regenerate(int p){
        table[p].clear();++stats[3];
        std::vector<std::vector<int>> choices(phy.C);
        for(int c=0;c<phy.C;++c)for(int u=0;u<phy.U;++u)
            if(legal[(c*phy.P+p)*phy.U+u]&&u!=action[c*phy.P+p])choices[c].push_back(u);
        for(int c=0;c<phy.C;++c)for(int u:choices[c])add(p,-1,c,u);
        for(int s=0;s<phy.C;++s)if(action[s*phy.P+p]>=0){
            add(p,s,-1,-1);
            for(int c=0;c<phy.C;++c)if(c!=s)for(int u:choices[c])add(p,s,c,u);
        }
    }
    void evaluate(Edit& e){
        std::vector<int64_t> old(phy.C);uint64_t oldmask=masks[e.prg];
        for(int c=0;c<phy.C;++c){old[c]=action[c*phy.P+e.prg];if(e.target[c]!=-2)action[c*phy.P+e.prg]=e.target[c];}
        mask(e.prg);e.dj=e.dd=0;++phy.stats[0];++stats[4];
        for(size_t i=0;i<e.users.size();++i){int u=e.users[i];auto& v=e.payload[i];
            phy.ue(u,action.data(),masks,v.data());
            e.dj+=quant(u,v[EXPECTED])-contribution[u];e.dd+=bytes(v[DEQUEUE])-bytes(values[u*FIELDS+DEQUEUE]);
        }
        for(int c=0;c<phy.C;++c)action[c*phy.P+e.prg]=old[c];masks[e.prg]=oldmask;e.valid=true;
    }
    std::vector<Edit*> flat(){
        std::vector<Edit*> out;for(auto& block:table)for(auto& e:block)out.push_back(&e);return out;
    }
    Edit* choose(const std::vector<Edit*>& all,int64_t threshold){
        int64_t maximum=INT64_MIN;
        for(auto e:all)if(dequeue+e->dd>=floor&&e->dj>threshold)maximum=std::max(maximum,e->dj);
        Edit* best=nullptr;
        for(auto e:all)if(dequeue+e->dd>=floor&&e->dj>threshold&&e->dj>=maximum-1)
            if(!best||e->key<best->key)best=e;
        return best;
    }
    void put(std::vector<double>& v,Edit* e){
        for(size_t i=0;i<e->users.size();++i)std::copy(e->payload[i].begin(),e->payload[i].end(),v.begin()+e->users[i]*FIELDS);
    }
    bool guard(const std::vector<Edit*>& selected){
        std::vector<double> single=values,batch=values;put(single,selected[0]);for(auto e:selected)put(batch,e);
        for(int u=0;u<phy.U;++u){
            if(single[u*FIELDS+COUNT]>0&&batch[u*FIELDS+COUNT]==0)return false;
            if(((weak>>u)&1)&&batch[u*FIELDS+EXPECTED]+1.e-6<single[u*FIELDS+EXPECTED])return false;
        }
        return true;
    }
    void observe(Observer cb,void* userdata,int iteration,const std::vector<Edit*>& all,const std::vector<Edit*>& selected,
                 int64_t fresh,int proposed,int rejected){
        if(!cb)return;
        std::vector<int64_t> keys,prgs,dj,dd,selectedkeys;
        std::vector<double> dense;
        keys.reserve(all.size());dense.reserve(all.size()*values.size());
        for(auto e:all){
            keys.push_back(e->key);prgs.push_back(e->prg);dj.push_back(e->dj);dd.push_back(e->dd);
            auto v=values;put(v,e);dense.insert(dense.end(),v.begin(),v.end());
        }
        for(auto e:selected)selectedkeys.push_back(e->key);
        int64_t meta[12]={iteration,int64_t(all.size()),score,dequeue,floor,int64_t(selected.size()),fresh,proposed,rejected,stats[2],stats[3],0};
        if(cb(userdata,meta,action.data(),values.data(),keys.data(),prgs.data(),dj.data(),dd.data(),dense.data(),selectedkeys.data()))
            throw std::runtime_error("Observer rejected gate");
    }
    void run(int maxrounds,Observer cb,void* userdata){
        auto t=Clock::now();for(int p=0;p<phy.P;++p)regenerate(p);
        if(profile)times[1]+=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        for(int iteration=1;iteration<=maxrounds;++iteration){
            auto start=Clock::now();auto all=flat();stats[5]+=all.size();int64_t fresh=0;
            auto stamp=Clock::now();
            for(auto e:all)if(mode==0||!e->valid){evaluate(*e);++fresh;}
            if(profile)times[2]+=std::chrono::duration<double,std::milli>(Clock::now()-stamp).count();
            stats[14]+=int64_t(all.size())-fresh;if(iteration==1)stats[8]=fresh;
            stamp=Clock::now();int64_t threshold=(iteration<=2?1:16)*SCALE;auto best=choose(all,threshold);
            std::vector<Edit*> selected;
            if(best)selected.push_back(best);
            int proposed=int(selected.size()),rejected=0;
            if(best&&mode>=2){
                std::vector<Edit*> remaining;
                for(auto e:all)if(e!=best&&e->dj>threshold&&dequeue+e->dd>=floor)remaining.push_back(e);
                // Integer deltas have a one-unit epsilon band. Build only the
                // necessary score bands, preserving the same canonical tie law.
                __uint128_t used=best->footprint;int64_t change=best->dd;
                while(!remaining.empty()&&selected.size()<4){
                    auto next=choose(remaining,threshold);if(!next)break;
                    remaining.erase(std::remove(remaining.begin(),remaining.end(),next),remaining.end());
                    if((used&next->footprint)||dequeue+change+next->dd<floor)continue;
                    selected.push_back(next);used|=next->footprint;change+=next->dd;
                }
                proposed=int(selected.size());
                if(mode==3&&selected.size()>1&&!guard(selected)){selected.resize(1);++stats[6];rejected=1;}
            }
            if(profile)times[3]+=std::chrono::duration<double,std::milli>(Clock::now()-stamp).count();
            observe(cb,userdata,iteration,all,selected,fresh,proposed,rejected);
            stats[0]=iteration;
            if(selected.empty()){
                if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(iteration>=3){stats[10]=1;break;}else continue;
            }
            __uint128_t changed=0;std::vector<int> changed_prgs;
            for(auto e:selected){
                for(int c=0;c<phy.C;++c)if(e->target[c]!=-2)action[c*phy.P+e->prg]=e->target[c];
                put(values,e);score+=e->dj;dequeue+=e->dd;changed|=e->support;changed_prgs.push_back(e->prg);
            }
            for(int u=0;u<phy.U;++u)if((changed>>u)&1)contribution[u]=quant(u,values[u*FIELDS+EXPECTED]);
            stats[1]+=selected.size();if(selected.size()>1)++stats[7];
            stamp=Clock::now();
            for(auto& block:table)for(auto& e:block)if(e.support&changed)e.valid=false;
            for(int p:changed_prgs){mask(p);regenerate(p);}
            if(profile)times[1]+=std::chrono::duration<double,std::milli>(Clock::now()-stamp).count();
            if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        }
        stats[11]=score;stats[12]=floor;stats[13]=dequeue;
    }
};
}

extern "C" {
int pd_solve(void* context,const uint8_t* legal,const double* weights,const int64_t* raw,const int64_t* warm,
             int mode,uint64_t weak,int maxrounds,int profile,int64_t* action,double* values,int64_t* stats,
             double* times,pd::Observer observer,void* userdata){
    try{
        auto start=pd::Clock::now();pd::Solver s(*static_cast<Context*>(context),legal,weights,raw,warm,mode,weak,profile!=0);
        if(profile)s.times[0]=std::chrono::duration<double,std::milli>(pd::Clock::now()-start).count();
        s.run(maxrounds,observer,userdata);
        std::copy(s.action.begin(),s.action.end(),action);std::copy(s.values.begin(),s.values.end(),values);
        std::copy(s.stats,s.stats+18,stats);std::copy(s.times,s.times+8,times);return 0;
    }catch(const std::exception& e){pd::error=e.what();return -1;}catch(...){pd::error="Unknown persistent error";return -2;}
}
const char* pd_error(){return pd::error.c_str();}
}
