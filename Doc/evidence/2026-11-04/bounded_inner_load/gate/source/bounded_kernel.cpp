// Separately versioned exact capacity-bound solver; frozen kernels unchanged.
#include "persistent_kernel.cpp"
#include <queue>

namespace lb {
using Observer=int(*)(void*,const int64_t*,const int64_t*,const double*,const int64_t*,
    const int64_t*,const int64_t*,const int64_t*,const int64_t*,const int64_t*);
struct Bound {int64_t upper=0,dd=0;bool valid=false;};
struct Node {pd::Edit* edit;Bound* bound;};
struct Less {
    bool operator()(const Node& a,const Node& b) const {
        if(a.bound->upper!=b.bound->upper)return a.bound->upper<b.bound->upper;
        return a.edit->key>b.edit->key;
    }
};
struct Best {
    int64_t maximum=INT64_MIN;
    pd::Edit* top=nullptr;pd::Edit* below=nullptr;
    static void take(pd::Edit*& old,pd::Edit* e){if(!old||e->key<old->key)old=e;}
    void add(pd::Edit* e,int64_t threshold){
        auto g=e->dj;if(g<=threshold)return;
        if(!top||g>maximum){below=top&&g==maximum+1?top:nullptr;top=e;maximum=g;}
        else if(g==maximum)take(top,e);
        else if(g==maximum-1)take(below,e);
    }
    pd::Edit* winner() const {return below&&below->key<top->key?below:top;}
};
struct Solver:pd::Solver {
    std::vector<std::vector<Bound>> bounds;
    std::vector<int> excluding;
    int64_t more[8]{}; // bound updates, heap pushes, fresh pruned, known reuse, floor pruned, threshold pruned
    Solver(Context& p,const uint8_t* l,const double* w,const int64_t* raw,const int64_t* warm,bool prof):
        pd::Solver(p,l,w,raw,warm,1,0,prof),bounds(p.P),excluding(p.U*p.P){
        for(int u=0;u<p.U;++u)exclusion(u);
    }
    void exclusion(int u){
        int first=5,second=5,n=0;
        for(int p=0;p<phy.P;++p)if(action[phy.serving[u]*phy.P+p]==u){
            int v=phy.caps[u*phy.P+p];
            if(v<first){second=first;first=v;n=1;}
            else if(v==first)++n;else second=std::min(second,v);
        }
        for(int p=0;p<phy.P;++p)excluding[u*phy.P+p]=
            action[phy.serving[u]*phy.P+p]==u&&phy.caps[u*phy.P+p]==first&&n==1?second:first;
    }
    void generate(int p){regenerate(p);bounds[p].assign(table[p].size(),Bound{});}
    void bound(pd::Edit& e,Bound& b){
        b.upper=b.dd=0;
        for(int u:e.users){
            int c=phy.serving[u];bool before=action[c*phy.P+e.prg]==u;
            bool after=(e.target[c]==-2?action[c*phy.P+e.prg]:e.target[c])==u;
            int count=int(values[u*FIELDS+COUNT])-int(before)+int(after);
            int layers=count?std::min(excluding[u*phy.P+e.prg],after?int(phy.caps[u*phy.P+e.prg]):5):0;
            if(count&&(layers<1||layers>4))throw std::runtime_error("Invalid logical layer");
            double d=std::min(phy.queue[u],double(phy.tbs[(u*(phy.P+1)+count)*5+layers]));
            b.upper+=quant(u,d)-contribution[u];b.dd+=bytes(d)-bytes(values[u*FIELDS+DEQUEUE]);
        }
        b.valid=true;++more[0];
    }
    void observe(Observer cb,void* userdata,int iteration,const std::vector<Node>& all,
        const std::vector<uint8_t>& before,const std::vector<uint8_t>& fresh,const Best& best){
        if(!cb)return;
        std::vector<int64_t> keys,upper,delta,dd,status;
        int64_t threshold=(iteration<=2?1:16)*pd::SCALE;auto winner=best.winner();
        for(size_t i=0;i<all.size();++i){auto n=all[i];auto e=n.edit;
            keys.push_back(e->key);upper.push_back(n.bound->upper);dd.push_back(n.bound->dd);
            delta.push_back(e->valid?e->dj:INT64_MIN);
            status.push_back(int(e->valid)|(int(fresh[i])<<1)|(int(before[i])<<2)|
                (int(dequeue+n.bound->dd>=floor)<<3));
        }
        int64_t selected=winner?winner->key:-1;
        int64_t meta[16]={iteration,int64_t(all.size()),score,dequeue,floor,winner?1:0,
            int64_t(std::count(fresh.begin(),fresh.end(),1)),
            int64_t(std::count(before.begin(),before.end(),1)),0,stats[2],stats[3],threshold,
            best.maximum,winner?winner->dj:0,0,0};
        if(cb(userdata,meta,action.data(),values.data(),keys.data(),upper.data(),delta.data(),dd.data(),status.data(),&selected))
            throw std::runtime_error("Bound observer rejected gate");
    }
    void run(int maxrounds,Observer cb,void* userdata){
        auto stamp=pd::Clock::now();for(int p=0;p<phy.P;++p)generate(p);
        if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
        for(int iteration=1;iteration<=maxrounds;++iteration){
            auto first=pd::Clock::now();std::vector<Node> all;all.reserve(2048);
            for(int p=0;p<phy.P;++p)for(size_t i=0;i<table[p].size();++i)all.push_back({&table[p][i],&bounds[p][i]});
            stats[5]+=all.size();int64_t threshold=(iteration<=2?1:16)*pd::SCALE;
            stamp=pd::Clock::now();
            for(auto n:all)if(!n.bound->valid)bound(*n.edit,*n.bound);
            if(profile)times[4]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            stamp=pd::Clock::now();Best best;std::vector<Node> heap;heap.reserve(all.size());
            std::vector<uint8_t> before(cb?all.size():0),fresh(cb?all.size():0);
            for(size_t i=0;i<all.size();++i){auto n=all[i];auto e=n.edit;
                if(cb)before[i]=e->valid;
                if(dequeue+n.bound->dd<floor){++more[4];continue;}
                if(e->valid){best.add(e,threshold);++more[3];continue;}
                if(n.bound->upper<=threshold){++more[5];continue;}
                heap.push_back(n);++more[1];
            }
            std::make_heap(heap.begin(),heap.end(),Less{});
            if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            int64_t paid=0;
            while(!heap.empty()){
                stamp=pd::Clock::now();auto n=heap.front();auto winner=best.winner();
                if(winner&&(n.bound->upper<best.maximum-1||
                    (n.bound->upper==best.maximum-1&&n.edit->key>=winner->key)))break;
                std::pop_heap(heap.begin(),heap.end(),Less{});heap.pop_back();
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                stamp=pd::Clock::now();evaluate(*n.edit);++paid;
                if(profile)times[2]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(n.edit->dj>n.bound->upper||n.edit->dd!=n.bound->dd)throw std::runtime_error("Invalid capacity/floor bound");
                if(cb)for(size_t i=0;i<all.size();++i)if(all[i].edit==n.edit){fresh[i]=1;break;}
                best.add(n.edit,threshold);
            }
            more[2]+=heap.size();stats[14]+=int64_t(all.size())-paid;
            if(iteration==1)stats[8]=paid;
            observe(cb,userdata,iteration,all,before,fresh,best);stats[0]=iteration;
            auto chosen=best.winner();
            if(!chosen){
                if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(pd::Clock::now()-first).count();
                if(iteration>=3){stats[10]=1;break;}else continue;
            }
            int p=chosen->prg;uint64_t changed=chosen->support;
            for(int c=0;c<phy.C;++c)if(chosen->target[c]!=-2)action[c*phy.P+p]=chosen->target[c];
            put(values,chosen);score+=chosen->dj;dequeue+=chosen->dd;++stats[1];
            stamp=pd::Clock::now();
            for(int u=0;u<phy.U;++u)if((changed>>u)&1){contribution[u]=quant(u,values[u*FIELDS+EXPECTED]);exclusion(u);}
            for(int q=0;q<phy.P;++q)for(size_t i=0;i<table[q].size();++i)if(table[q][i].support&changed){
                table[q][i].valid=false;bounds[q][i].valid=false;
            }
            mask(p);generate(p);
            if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(pd::Clock::now()-first).count();
        }
        stats[11]=score;stats[12]=floor;stats[13]=dequeue;
    }
};
}
extern "C" {
int lb_solve(void* context,const uint8_t* legal,const double* weights,const int64_t* raw,const int64_t* warm,
    int maxrounds,int profile,int64_t* action,double* values,int64_t* stats,int64_t* more,double* times,
    lb::Observer observer,void* userdata){
    try{
        auto start=pd::Clock::now();lb::Solver s(*static_cast<Context*>(context),legal,weights,raw,warm,profile!=0);
        if(profile)s.times[0]=std::chrono::duration<double,std::milli>(pd::Clock::now()-start).count();
        s.run(maxrounds,observer,userdata);
        std::copy(s.action.begin(),s.action.end(),action);std::copy(s.values.begin(),s.values.end(),values);
        std::copy(s.stats,s.stats+18,stats);std::copy(s.more,s.more+8,more);std::copy(s.times,s.times+8,times);return 0;
    }catch(const std::exception& e){pd::error=e.what();return -1;}catch(...){pd::error="Unknown bounded error";return -2;}
}
const char* lb_error(){return pd::error.c_str();}
}
