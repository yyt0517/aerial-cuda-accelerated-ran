// Exact two-level tournament: one global leaf per PRG, sparse dirty updates.
// The frozen physical arithmetic and B scan kernel are included, never edited.
#include "bounded_kernel.cpp"

namespace ts {
struct Entry {pd::Edit* e=nullptr;lb::Bound* b=nullptr;bool feasible=false;int dirty_epoch=-1,paid_epoch=-1;};
static Entry* priority(Entry* a,Entry* b){
    if(!a)return b;if(!b)return a;
    if(a->b->upper!=b->b->upper)return a->b->upper>b->b->upper?a:b;
    return a->e->key<b->e->key?a:b;
}
static lb::Best combine(const lb::Best& a,const lb::Best& b){
    lb::Best v;
    if(a.top)v.add(a.top,INT64_MIN);if(a.below)v.add(a.below,INT64_MIN);
    if(b.top)v.add(b.top,INT64_MIN);if(b.below)v.add(b.below,INT64_MIN);
    return v;
}
struct Block {
    int size=1,version=0;std::vector<Entry> entries;
    std::vector<Entry*> unknown;std::vector<lb::Best> exact;
    int feasible_count=0,known_count=0;
    void leaf(int i){
        auto& n=entries[i];int k=size+i;unknown[k]=nullptr;exact[k]=lb::Best{};
        if(n.feasible){if(n.e->valid)exact[k].add(n.e,INT64_MIN);else unknown[k]=&n;}
    }
    void update(int i){
        int k=size+i;
        bool oldf=unknown[k]||exact[k].top;bool oldk=exact[k].top!=nullptr;
        leaf(i);feasible_count+=int(entries[i].feasible)-int(oldf);
        known_count+=int(exact[k].top!=nullptr)-int(oldk);
        for(k/=2;k;k/=2){unknown[k]=priority(unknown[2*k],unknown[2*k+1]);exact[k]=combine(exact[2*k],exact[2*k+1]);}
    }
    void build(){
        size=1;while(size<int(entries.size()))size*=2;
        unknown.assign(2*size,nullptr);exact.assign(2*size,lb::Best{});feasible_count=known_count=0;
        for(size_t i=0;i<entries.size();++i){leaf(i);feasible_count+=entries[i].feasible;known_count+=entries[i].feasible&&entries[i].e->valid;}
        for(int k=size-1;k;--k){unknown[k]=priority(unknown[2*k],unknown[2*k+1]);exact[k]=combine(exact[2*k],exact[2*k+1]);}
    }
};
struct Reference {int prg,index,version;};
struct Solver:lb::Solver {
    std::vector<Block> blocks;std::vector<std::vector<Reference>> dependent;
    std::vector<Entry*> unknown_tree;std::vector<lb::Best> exact_tree;
    int global_size=1;int64_t total=0,tree_updates=0,priority_queries=0,dependency_visits=0;
    Solver(Context& p,const uint8_t* l,const double* w,const int64_t* raw,const int64_t* warm,bool prof):
        lb::Solver(p,l,w,raw,warm,prof),blocks(p.P),dependent(p.U){
        while(global_size<p.P)global_size*=2;
        unknown_tree.assign(2*global_size,nullptr);exact_tree.assign(2*global_size,lb::Best{});
    }
    void global(int p){
        int k=global_size+p;unknown_tree[k]=blocks[p].unknown[1];exact_tree[k]=blocks[p].exact[1];
        for(k/=2;k;k/=2){unknown_tree[k]=priority(unknown_tree[2*k],unknown_tree[2*k+1]);exact_tree[k]=combine(exact_tree[2*k],exact_tree[2*k+1]);}
        ++tree_updates;
    }
    void regenerate_block(int p){
        auto& block=blocks[p];total-=block.entries.size();++block.version;
        generate(p);block.entries.clear();block.entries.reserve(table[p].size());
        for(size_t i=0;i<table[p].size();++i){
            auto& e=table[p][i];auto& b=bounds[p][i];bound(e,b);
            block.entries.push_back({&e,&b,dequeue+b.dd>=floor,-1,-1});
            for(int u:e.users)dependent[u].push_back({p,int(i),block.version});
        }
        total+=block.entries.size();block.build();global(p);
    }
    pd::Edit* winner(int64_t threshold){
        auto& v=exact_tree[1];if(!v.top||v.maximum<=threshold)return nullptr;
        return v.below&&v.below->dj>threshold&&v.below->key<v.top->key?v.below:v.top;
    }
    void run(int maxrounds,lb::Observer cb,void* userdata){
        auto stamp=pd::Clock::now();for(int p=0;p<phy.P;++p)regenerate_block(p);
        if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
        for(int iteration=1;iteration<=maxrounds;++iteration){
            auto first=pd::Clock::now();stats[5]+=total;int64_t threshold=(iteration<=2?1:16)*pd::SCALE;
            std::vector<lb::Node> all;std::vector<uint8_t> before,fresh;
            if(cb){
                all.reserve(total);before.reserve(total);
                for(auto& block:blocks)for(auto& n:block.entries){all.push_back({n.e,n.b});before.push_back(n.e->valid);}
            }
            int64_t paid=0;
            while(true){
                if(profile)stamp=pd::Clock::now();auto n=unknown_tree[1];auto chosen=winner(threshold);++priority_queries;
                if(!n||n->b->upper<=threshold)break;
                auto maximum=exact_tree[1].maximum;
                if(chosen&&(n->b->upper<maximum-1||(n->b->upper==maximum-1&&n->e->key>=chosen->key)))break;
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(profile)stamp=pd::Clock::now();evaluate(*n->e);++paid;n->paid_epoch=iteration;
                if(profile)times[2]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(n->e->dj>n->b->upper||n->e->dd!=n->b->dd)throw std::runtime_error("Invalid tournament capacity bound");
                if(profile)stamp=pd::Clock::now();int p=n->e->prg;int i=int(n-blocks[p].entries.data());
                blocks[p].update(i);global(p);
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            }
            stats[0]=iteration;if(iteration==1)stats[8]=paid;
            auto chosen=winner(threshold);
            if(cb){
                for(auto& block:blocks)for(auto& n:block.entries)fresh.push_back(n.paid_epoch==iteration);
                lb::Best qualifying;
                if(exact_tree[1].top)qualifying.add(exact_tree[1].top,threshold);
                if(exact_tree[1].below)qualifying.add(exact_tree[1].below,threshold);
                observe(cb,userdata,iteration,all,before,fresh,qualifying);
            }
            if(!chosen){
                if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(pd::Clock::now()-first).count();
                if(iteration>=3){stats[10]=1;break;}else continue;
            }
            int changed_prg=chosen->prg;uint64_t changed=chosen->support;int64_t previous_d=dequeue;
            for(int c=0;c<phy.C;++c)if(chosen->target[c]!=-2)action[c*phy.P+changed_prg]=chosen->target[c];
            put(values,chosen);score+=chosen->dj;dequeue+=chosen->dd;++stats[1];
            if(profile)stamp=pd::Clock::now();
            for(int u=0;u<phy.U;++u)if((changed>>u)&1){contribution[u]=quant(u,values[u*FIELDS+EXPECTED]);exclusion(u);}
            if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            std::vector<bool> touched(phy.P,false);
            for(int u=0;u<phy.U;++u)if((changed>>u)&1)for(auto ref:dependent[u]){
                ++dependency_visits;
                auto& block=blocks[ref.prg];
                if(ref.prg==changed_prg||ref.version!=block.version)continue;
                auto& n=block.entries[ref.index];if(n.dirty_epoch==iteration)continue;n.dirty_epoch=iteration;
                n.e->valid=false;n.b->valid=false;
                if(profile)stamp=pd::Clock::now();bound(*n.e,*n.b);
                if(profile)times[4]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(profile)stamp=pd::Clock::now();n.feasible=dequeue+n.b->dd>=floor;block.update(ref.index);touched[ref.prg]=true;
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            }
            // The floor is global: a root-D change can change eligibility even
            // for unchanged UE support. Refresh only leaves crossing the cut;
            // never assume that dirty-PHY dependencies cover this condition.
            if(previous_d!=dequeue)for(int p=0;p<phy.P;++p)if(p!=changed_prg){
                auto& block=blocks[p];
                for(size_t i=0;i<block.entries.size();++i){auto& n=block.entries[i];bool f=dequeue+n.b->dd>=floor;
                    if(f!=n.feasible){n.feasible=f;block.update(i);touched[p]=true;}
                }
            }
            if(profile)stamp=pd::Clock::now();mask(changed_prg);regenerate_block(changed_prg);
            for(int p=0;p<phy.P;++p)if(touched[p])global(p);
            if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            if(profile&&iteration==1)times[6]=std::chrono::duration<double,std::milli>(pd::Clock::now()-first).count();
        }
        stats[11]=score;stats[12]=floor;stats[13]=dequeue;
        more[1]=tree_updates;more[2]=priority_queries;more[6]=dependency_visits;
    }
};
}
extern "C" {
int ts_solve(void* context,const uint8_t* legal,const double* weights,const int64_t* raw,const int64_t* warm,
    int maxrounds,int profile,int64_t* action,double* values,int64_t* stats,int64_t* more,double* times,
    lb::Observer observer,void* userdata){
    try{
        auto start=pd::Clock::now();ts::Solver s(*static_cast<Context*>(context),legal,weights,raw,warm,profile!=0);
        if(profile)s.times[0]=std::chrono::duration<double,std::milli>(pd::Clock::now()-start).count();
        s.run(maxrounds,observer,userdata);
        std::copy(s.action.begin(),s.action.end(),action);std::copy(s.values.begin(),s.values.end(),values);
        std::copy(s.stats,s.stats+18,stats);std::copy(s.more,s.more+8,more);std::copy(s.times,s.times+8,times);return 0;
    }catch(const std::exception& e){pd::error=e.what();return -1;}catch(...){pd::error="Unknown tournament error";return -2;}
}
const char* ts_error(){return pd::error.c_str();}
}
