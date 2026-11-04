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
    std::vector<Entry*> unknown;
    void leaf(int i){
        auto& n=entries[i];int k=size+i;unknown[k]=n.feasible&&!n.e->valid?&n:nullptr;
    }
    void update(int i){
        int k=size+i;
        leaf(i);
        for(k/=2;k;k/=2)unknown[k]=priority(unknown[2*k],unknown[2*k+1]);
    }
    void build(){
        size=1;while(size<int(entries.size()))size*=2;
        unknown.assign(2*size,nullptr);
        for(size_t i=0;i<entries.size();++i)leaf(i);
        for(int k=size-1;k;--k)unknown[k]=priority(unknown[2*k],unknown[2*k+1]);
    }
};
struct Reference {int prg,index,version;};
struct Solver:lb::Solver {
    std::vector<Block> blocks;std::vector<std::vector<Reference>> dependent;
    std::vector<Entry*> unknown_tree;lb::Best current;
    bool direct_gamma=false;size_t mask_count=0;int64_t direct_hits=0;
    std::vector<std::array<double,4>> direct_values;std::vector<uint8_t> direct_valid;
    int global_size=1;int64_t total=0,tree_updates=0,priority_queries=0,dependency_visits=0;
    Solver(Context& p,const uint8_t* l,const double* w,const int64_t* raw,const int64_t* warm,bool prof,bool fast=false):
        lb::Solver(p,l,w,raw,warm,prof),blocks(p.P),dependent(p.U){
        while(global_size<p.P)global_size*=2;
        unknown_tree.assign(2*global_size,nullptr);
        direct_gamma=fast&&p.C<=8;
        if(direct_gamma){mask_count=size_t(1)<<p.C;direct_values.resize(p.U*p.P*mask_count);direct_valid.assign(direct_values.size(),0);}
    }
    const std::array<double,4>& gamma(int u,int p,uint64_t mask){
        if(!direct_gamma)return phy.gamma(u,p,mask);
        mask&=~(uint64_t(1)<<phy.serving[u]);size_t i=(u*phy.P+p)*mask_count+mask;
        if(direct_valid[i]){++phy.stats[2];++direct_hits;return direct_values[i];}
        // A cache miss still calls the ORIGINAL frozen matrix computation.
        direct_values[i]=phy.gamma(u,p,mask);direct_valid[i]=1;return direct_values[i];
    }
    void ue(int u,double* out){
        ++phy.stats[1];int count=0,layers=4;
        for(int p=0;p<phy.P;++p)if(action[phy.serving[u]*phy.P+p]==u){++count;layers=std::min(layers,int(phy.caps[u*phy.P+p]));}
        std::fill(out,out+FIELDS,0.);out[COUNT]=count;
        if(!count){out[LAYERS]=-1.;out[SINR]=-999.;out[BLER]=-1.;return;}
        double total=0.,interference=0.;
        for(int p=0;p<phy.P;++p)if(action[phy.serving[u]*phy.P+p]==u){
            const auto& g=gamma(u,p,masks[p]);
            for(int l=0;l<layers;++l)total+=g[l];
            for(int c=0;c<phy.C;++c)if(c!=phy.serving[u]&&((masks[p]>>c)&1))interference+=phy.traces[(u*phy.P+p)*phy.C+c];
        }
        out[LAYERS]=layers;out[CAPACITY]=phy.tbs[(u*(phy.P+1)+count)*5+layers];
        out[DEQUEUE]=std::min(phy.queue[u],out[CAPACITY]);
        out[SINR]=10.*std::log10(std::max(total/(count*layers),1.e-30));
        out[BLER]=phy.lookup(phy.mcs[u],out[SINR]);out[EXPECTED]=out[DEQUEUE]*(1.-out[BLER]);out[INTERFERENCE]=interference/count;
    }
    void evaluate(pd::Edit& e){
        if(!direct_gamma){pd::Solver::evaluate(e);return;}
        std::array<int64_t,63> old{};uint64_t oldmask=masks[e.prg];
        for(int c=0;c<phy.C;++c){old[c]=action[c*phy.P+e.prg];if(e.target[c]!=-2)action[c*phy.P+e.prg]=e.target[c];}
        mask(e.prg);e.dj=e.dd=0;++phy.stats[0];++stats[4];
        for(size_t i=0;i<e.users.size();++i){int u=e.users[i];auto& v=e.payload[i];ue(u,v.data());
            e.dj+=quant(u,v[EXPECTED])-contribution[u];e.dd+=bytes(v[DEQUEUE])-bytes(values[u*FIELDS+DEQUEUE]);}
        for(int c=0;c<phy.C;++c)action[c*phy.P+e.prg]=old[c];masks[e.prg]=oldmask;e.valid=true;
    }
    void global(int p){
        int k=global_size+p;unknown_tree[k]=blocks[p].unknown[1];
        for(k/=2;k;k/=2)unknown_tree[k]=priority(unknown_tree[2*k],unknown_tree[2*k+1]);
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
        auto& v=current;if(!v.top||v.maximum<=threshold)return nullptr;
        return v.below&&v.below->dj>threshold&&v.below->key<v.top->key?v.below:v.top;
    }
    void run(int maxrounds,lb::Observer cb,void* userdata){
        auto stamp=pd::Clock::now();for(int p=0;p<phy.P;++p)regenerate_block(p);
        if(profile)times[1]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
        for(int iteration=1;iteration<=maxrounds;++iteration){
            auto first=pd::Clock::now();stats[5]+=total;int64_t threshold=(iteration<=2?1:16)*pd::SCALE;
            std::vector<lb::Node> all;std::vector<uint8_t> before,fresh;
            if(profile)stamp=pd::Clock::now();current=lb::Best{};
            for(auto& block:blocks)for(auto& n:block.entries)if(n.feasible&&n.e->valid)current.add(n.e,threshold);
            if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            if(cb){
                all.reserve(total);before.reserve(total);
                for(auto& block:blocks)for(auto& n:block.entries){all.push_back({n.e,n.b});before.push_back(n.e->valid);}
            }
            int64_t paid=0;
            while(true){
                if(profile)stamp=pd::Clock::now();auto n=unknown_tree[1];auto chosen=winner(threshold);++priority_queries;
                if(!n||n->b->upper<=threshold)break;
                auto maximum=current.maximum;
                if(chosen&&(n->b->upper<maximum-1||(n->b->upper==maximum-1&&n->e->key>=chosen->key)))break;
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(profile)stamp=pd::Clock::now();evaluate(*n->e);++paid;n->paid_epoch=iteration;
                if(profile)times[2]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
                if(n->e->dj>n->b->upper||n->e->dd!=n->b->dd)throw std::runtime_error("Invalid tournament capacity bound");
                current.add(n->e,threshold);
                if(profile)stamp=pd::Clock::now();int p=n->e->prg;int i=int(n-blocks[p].entries.data());
                blocks[p].update(i);global(p);
                if(profile)times[3]+=std::chrono::duration<double,std::milli>(pd::Clock::now()-stamp).count();
            }
            stats[0]=iteration;if(iteration==1)stats[8]=paid;
            auto chosen=winner(threshold);
            if(cb){
                for(auto& block:blocks)for(auto& n:block.entries)fresh.push_back(n.paid_epoch==iteration);
                observe(cb,userdata,iteration,all,before,fresh,current);
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
        more[1]=tree_updates;more[2]=priority_queries;more[6]=dependency_visits;more[7]=direct_hits;
    }
};
}
extern "C" {
int ts_solve(void* context,const uint8_t* legal,const double* weights,const int64_t* raw,const int64_t* warm,
    int maxrounds,int profile,int fast,int64_t* action,double* values,int64_t* stats,int64_t* more,double* times,
    lb::Observer observer,void* userdata){
    try{
        auto start=pd::Clock::now();ts::Solver s(*static_cast<Context*>(context),legal,weights,raw,warm,profile!=0,fast!=0);
        if(profile)s.times[0]=std::chrono::duration<double,std::milli>(pd::Clock::now()-start).count();
        s.run(maxrounds,observer,userdata);
        std::copy(s.action.begin(),s.action.end(),action);std::copy(s.values.begin(),s.values.end(),values);
        std::copy(s.stats,s.stats+18,stats);std::copy(s.more,s.more+8,more);std::copy(s.times,s.times+8,times);return 0;
    }catch(const std::exception& e){pd::error=e.what();return -1;}catch(...){pd::error="Unknown tournament error";return -2;}
}
const char* ts_error(){return pd::error.c_str();}
int ts_prepare(int U,int C,int P,int slot_count,double cap_db,const double* metadata,const double* lut,
    const int64_t* supplied,int provided,const int64_t* serving,const int64_t* caps,const double* queue,
    const int64_t* full_tbs,const uint8_t* available,const int64_t* slots,const int64_t* allocation,
    int64_t* mcs,int64_t* tbs,uint8_t* legal,int64_t* raw){
    try{
        if(U<=0||C<=0||C>63||P<=0||U+P>63||slot_count<0||!std::isfinite(cap_db))throw std::runtime_error("Invalid input dimensions");
        uint64_t key=P;
        for(int c=0;c<C;++c){if(key>uint64_t(INT64_MAX)/(U+3))throw std::runtime_error("Unsafe canonical key");key*=U+3;}
        for(int u=0;u<U;++u){
            if(serving[u]<0||serving[u]>=C||!std::isfinite(queue[u])||queue[u]<0||queue[u]!=std::floor(queue[u]))
                throw std::runtime_error("Invalid queue/association");
            for(int p=0;p<P;++p)if(caps[u*P+p]<1||caps[u*P+p]>4)throw std::runtime_error("Invalid layer cap");
            int m=0;
            if(provided)m=supplied[u];else{
                for(int i=0;i<5;++i)if(!std::isfinite(metadata[i*U+u]))throw std::runtime_error("Invalid MCS metadata");
                double prior=metadata[4*U+u];if(prior!=-1&&prior!=0&&prior!=1)throw std::runtime_error("Invalid previous CRC");
                float delta=float(metadata[U+u]);
                if(prior==1)delta=float(delta+float(metadata[2*U+u]));
                else if(prior==0)delta=float(delta-float(metadata[3*U+u]));
                float capped=std::min(float(metadata[u]),float(cap_db));float effective=float(capped-delta);
                for(int candidate=27;candidate>=0;--candidate)if(effective>=float(lut[candidate])){m=candidate;break;}
            }
            if(m<0||m>27)throw std::runtime_error("Invalid MCS");mcs[u]=m;
            std::copy(full_tbs+m*(P+1)*5,full_tbs+(m+1)*(P+1)*5,tbs+u*(P+1)*5);
        }
        for(int i=0;i<slot_count;++i)if(slots[i]<-1||slots[i]>=U)throw std::runtime_error("Invalid PFQ UE slot");
        for(int c=0;c<C;++c)for(int p=0;p<P;++p){
            int cp=c*P+p;for(int u=0;u<U;++u)legal[cp*U+u]=available[cp]&&serving[u]==c&&queue[u]>0;
            int64_t slot=allocation[p*C+c];if(slot<-1||slot>=slot_count)throw std::runtime_error("Invalid PFQ allocation");
            int64_t u=slot<0?-1:slots[slot];if(slot>=0&&(u<0||!legal[cp*U+u]))throw std::runtime_error("Illegal PFQ action");raw[cp]=u;
        }
        return 0;
    }catch(const std::exception& e){pd::error=e.what();return -1;}
}
}
