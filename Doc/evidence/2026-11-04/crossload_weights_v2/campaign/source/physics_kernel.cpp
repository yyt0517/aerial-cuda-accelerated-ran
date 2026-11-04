// Exact affected-UE evaluation under the existing activity-only 4T4R model.
// No global activity bank: MMSE results are cached only when requested.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix<Complex, 4, 4, Eigen::RowMajor>;
constexpr int FIELDS = 8;
enum Field { COUNT, LAYERS, CAPACITY, DEQUEUE, SINR, BLER, EXPECTED, INTERFERENCE };

struct Context {
    int U, C, P;
    double noise;
    std::vector<int64_t> serving, mcs, caps, tbs, offsets;
    std::vector<double> queue, snr, bler, traces;
    std::vector<Matrix> covariance, signal;
    std::vector<std::unordered_map<uint64_t, std::array<double, 4>>> cache;
    std::array<int64_t, 5> stats{}; // candidates, UE evaluations, gamma requests/misses, context bytes (approx.)

    Context(int u, int c, int p, double n, const int64_t* s, const int64_t* m,
            const int64_t* cap, const int64_t* tb, const double* q,
            const double* cov, const double* sig, const int64_t* off,
            const double* xs, const double* ys): U(u), C(c), P(p), noise(n),
            serving(s, s+u), mcs(m,m+u), caps(cap,cap+u*p), tbs(tb,tb+u*(p+1)*5),
            offsets(off,off+29), queue(q,q+u), snr(xs,xs+off[28]), bler(ys,ys+off[28]),
            traces(u*p*c), covariance(u*p*c), signal(u*p), cache(u*p) {
        for (int i=0; i<U*P*C; ++i) {
            for (int j=0; j<16; ++j) covariance[i].data()[j]=Complex(cov[32*i+2*j],cov[32*i+2*j+1]);
            traces[i]=covariance[i].trace().real();
        }
        for (int i=0; i<U*P; ++i)
            for (int j=0; j<16; ++j) signal[i].data()[j]=Complex(sig[32*i+2*j],sig[32*i+2*j+1]);
    }

    const std::array<double,4>& gamma(int u, int p, uint64_t mask) {
        mask &= ~(uint64_t(1)<<serving[u]);
        auto& entries=cache[u*P+p];
        ++stats[2];
        auto it=entries.find(mask);
        if (it!=entries.end()) return it->second;
        Matrix r=Matrix::Identity()*noise;
        for (int c=0;c<C;++c) if ((mask>>c)&1) r+=covariance[(u*P+p)*C+c];
        const Matrix& s=signal[u*P+p];
        Matrix e=(Matrix::Identity()+s.adjoint()*r.ldlt().solve(s)).inverse();
        std::array<double,4> g;
        for (int l=0;l<4;++l) g[l]=std::max(0.,1./e(l,l).real()-1.);
        ++stats[3];
        return entries.emplace(mask,g).first->second;
    }

    double lookup(int m,double db) const {
        // Match the native descending scan, including the nonmonotone MCS3 row.
        db=std::max(db,-4.75);
        int begin=offsets[m], end=offsets[m+1];
        if(db<snr[end-1]) return 1.;
        if(db>snr[begin]) return 0.;
        for(int j=end-1;j>begin;--j) if(db<=snr[j-1]) {
            double f=(db-snr[j])/(snr[j-1]-snr[j]);
            return bler[j]+f*(bler[j-1]-bler[j]);
        }
        return 0.;
    }

    void ue(int u,const int64_t* a,const std::vector<uint64_t>& masks,double* out) {
        ++stats[1];
        int count=0, layers=4;
        for(int p=0;p<P;++p) if(a[serving[u]*P+p]==u) {++count; layers=std::min(layers,int(caps[u*P+p]));}
        std::fill(out,out+FIELDS,0.);
        out[COUNT]=count;
        if(!count) {out[LAYERS]=-1.;out[SINR]=-999.;out[BLER]=-1.;return;}
        double total=0., interference=0.;
        for(int p=0;p<P;++p) if(a[serving[u]*P+p]==u) {
            const auto& g=gamma(u,p,masks[p]);
            for(int l=0;l<layers;++l) total+=g[l];
            for(int c=0;c<C;++c) if(c!=serving[u] && ((masks[p]>>c)&1)) interference+=traces[(u*P+p)*C+c];
        }
        out[LAYERS]=layers;
        out[CAPACITY]=tbs[(u*(P+1)+count)*5+layers];
        out[DEQUEUE]=std::min(queue[u],out[CAPACITY]);
        out[SINR]=10.*std::log10(std::max(total/(count*layers),1.e-30));
        out[BLER]=lookup(mcs[u],out[SINR]);
        out[EXPECTED]=out[DEQUEUE]*(1.-out[BLER]);
        out[INTERFERENCE]=interference/count;
    }

    void evaluate(const int64_t* a,const int64_t* base,const double* prediction,double* out) {
        ++stats[0];
        std::vector<uint64_t> masks(P,0);
        std::vector<bool> affected(U,base==nullptr);
        for(int p=0;p<P;++p) {
            bool activity_changed=false;
            for(int c=0;c<C;++c) {
                int64_t v=a[c*P+p];
                if(v>=0) masks[p]|=uint64_t(1)<<c;
                if(base && v!=base[c*P+p]) {
                    if(v>=0) affected[v]=true;
                    if(base[c*P+p]>=0) affected[base[c*P+p]]=true;
                    activity_changed|=(v>=0)!=(base[c*P+p]>=0);
                }
            }
            if(activity_changed) for(int c=0;c<C;++c) {
                // All physical neighbors are included, irrespective of proposal-graph sparsity.
                if(a[c*P+p]>=0) affected[a[c*P+p]]=true;
                if(base[c*P+p]>=0) affected[base[c*P+p]]=true;
            }
        }
        for(int u=0;u<U;++u) {
            if(affected[u]) ue(u,a,masks,out+u*FIELDS);
            else std::copy(prediction+u*FIELDS,prediction+(u+1)*FIELDS,out+u*FIELDS);
        }
    }
};

extern "C" {
void* lns_create(int u,int c,int p,double n,const int64_t* s,const int64_t* m,const int64_t* cap,
    const int64_t* tb,const double* q,const double* cov,const double* sig,const int64_t* off,
    const double* xs,const double* ys) {
    try { return new Context(u,c,p,n,s,m,cap,tb,q,cov,sig,off,xs,ys); } catch(...) { return nullptr; }
}
void lns_destroy(void* ptr) { delete static_cast<Context*>(ptr); }
int lns_evaluate(void* ptr,int batch,const int64_t* actions,const int64_t* base,const double* prediction,double* out) {
    try {
        auto& ctx=*static_cast<Context*>(ptr);
        for(int b=0;b<batch;++b) ctx.evaluate(actions+b*ctx.C*ctx.P,base,prediction,out+b*ctx.U*FIELDS);
        return 0;
    } catch(...) { return -1; }
}
void lns_stats(void* ptr,int64_t* out) {
    auto& ctx=*static_cast<Context*>(ptr);
    std::copy(ctx.stats.begin(),ctx.stats.end(),out);
}
}
