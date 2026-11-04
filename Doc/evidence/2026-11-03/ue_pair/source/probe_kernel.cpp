// Bounded diagnostic extensions; original physics_kernel.cpp remains unchanged.
#include "physics_kernel.cpp"

extern "C" {
int probe_prepare(void* ptr) {
    try {
        auto& c=*static_cast<Context*>(ptr);
        if(c.C>3) return 1;
        for(int u=0;u<c.U;++u) for(int p=0;p<c.P;++p)
            for(uint64_t m=0;m<(uint64_t(1)<<c.C);++m)
                if(!((m>>c.serving[u])&1)) c.gamma(u,p,m);
        return 0;
    } catch(...) {return 2;}
}

// Six affected-UE slots, each with 12 causal features; no candidate BLER lookup.
// Upper bounds and logical dequeue do not require the spatial/gamma bank.
int probe_features(void* ptr,int N,const int64_t* a,const int64_t* root,
                   const double* values,const double* weights,int spatial,float* x,double* upper,
                   double* total_d,double* proxy) {
    try {
        auto& c=*static_cast<Context*>(ptr);
        if(c.C>3) return 1;
        std::vector<std::array<int,5>> caps(c.U);
        std::vector<std::array<double,4>> sums(c.U);
        std::vector<uint64_t> masks(c.P,0);
        for(int p=0;p<c.P;++p) for(int cell=0;cell<c.C;++cell)
            if(root[cell*c.P+p]>=0) masks[p]|=uint64_t(1)<<cell;
        double root_j=0,root_d=0;
        for(int u=0;u<c.U;++u) {
            root_j+=weights[u]*values[u*FIELDS+EXPECTED];
            root_d+=values[u*FIELDS+DEQUEUE];
            for(int p=0;p<c.P;++p) if(root[c.serving[u]*c.P+p]==u) {
                ++caps[u][c.caps[u*c.P+p]];
                if(spatial) {
                    const auto& g=c.gamma(u,p,masks[p]);
                    for(int l=0;l<4;++l) sums[u][l]+=g[l];
                }
            }
        }
        for(int b=0;b<N;++b) {
            const auto* row=a+b*c.C*c.P;
            int prg=-1,edits=0;
            uint64_t mask=0;
            for(int cell=0;cell<c.C;++cell) for(int p=0;p<c.P;++p)
                if(row[cell*c.P+p]!=root[cell*c.P+p]) {
                    if(prg>=0&&prg!=p) return 3;
                    prg=p; ++edits;
                }
            if(edits>3) return 4;
            if(prg<0) prg=0;
            for(int cell=0;cell<c.C;++cell) if(row[cell*c.P+prg]>=0) mask|=uint64_t(1)<<cell;
            std::vector<int> affected;
            auto add=[&](int u) {
                if(u>=0&&std::find(affected.begin(),affected.end(),u)==affected.end()) affected.push_back(u);
            };
            // Semantic cell/donor/recipient ordering, independent of UE labels.
            for(int cell=0;cell<c.C;++cell) {
                int old=root[cell*c.P+prg],now=row[cell*c.P+prg];
                if(old!=now) {add(old);add(now);}
            }
            if(mask!=masks[prg]) for(int cell=0;cell<c.C;++cell) add(row[cell*c.P+prg]);
            if(affected.size()>6) return 5;
            upper[b]=root_j; total_d[b]=root_d; proxy[b]=0;
            std::fill(x+b*72,x+(b+1)*72,0.f);
            for(size_t slot=0;slot<affected.size();++slot) {
                int u=affected[slot],cell=c.serving[u],cap=c.caps[u*c.P+prg];
                bool old=root[cell*c.P+prg]==u,now=row[cell*c.P+prg]==u;
                auto histogram=caps[u]; histogram[cap]+=int(now)-int(old);
                int count=int(values[u*FIELDS+COUNT])+int(now)-int(old),layers=0;
                if(count) for(int l=1;l<=4;++l) if(histogram[l]) {layers=l;break;}
                double d=std::min(c.queue[u],double(c.tbs[(u*(c.P+1)+count)*5+layers]));
                double old_d=values[u*FIELDS+DEQUEUE],old_j=values[u*FIELDS+EXPECTED];
                upper[b]+=weights[u]*(d-old_j);
                total_d[b]+=d-old_d;
                double reliability=values[u*FIELDS+COUNT]>0?1-values[u*FIELDS+BLER]:.5;
                proxy[b]+=weights[u]*(d*reliability-old_j);
                auto* f=x+b*72+slot*12;
                f[0]=1; f[1]=float(count)/c.P; f[2]=float(layers)/4;
                f[3]=float(c.mcs[u])/27; f[4]=d/3000.; f[5]=weights[u]/4;
                f[6]=old_d/3000.; f[7]=values[u*FIELDS+COUNT]>0?values[u*FIELDS+BLER]:.5;
                if(spatial&&count) {
                    auto sum=sums[u];
                    if(old) {const auto& g=c.gamma(u,prg,masks[prg]);for(int l=0;l<4;++l)sum[l]-=g[l];}
                    if(now) {const auto& g=c.gamma(u,prg,mask);for(int l=0;l<4;++l)sum[l]+=g[l];}
                    for(int l=0;l<4;++l) f[8+l]=l<layers?std::log1p(std::max(0.,sum[l]/count))/5.:0.;
                }
            }
            // Conservative rounding guard; no BLER monotonicity assumption.
            upper[b]+=1.e-8*(1+std::abs(upper[b]));
        }
        return 0;
    } catch(...) {return 9;}
}
}
