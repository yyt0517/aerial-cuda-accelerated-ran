// Last full-table experiment: SoA, eight independent keys per AVX-512 lane.
// No FMA, reciprocal approximation, matrix-identity change or Woodbury update.
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
using V=__m512d;
struct Z {V r,i;};
static inline V number(double v){return _mm512_set1_pd(v);}
static inline Z zero(){return {number(0),number(0)};}
static inline Z add(Z a,Z b){return {_mm512_add_pd(a.r,b.r),_mm512_add_pd(a.i,b.i)};}
static inline Z sub(Z a,Z b){return {_mm512_sub_pd(a.r,b.r),_mm512_sub_pd(a.i,b.i)};}
static inline Z neg(Z a){return {_mm512_xor_pd(a.r,number(-0.)),_mm512_xor_pd(a.i,number(-0.))};}
static inline Z conj(Z a){return {a.r,_mm512_xor_pd(a.i,number(-0.))};}
static inline Z mul(Z a,Z b){return {_mm512_sub_pd(_mm512_mul_pd(a.r,b.r),_mm512_mul_pd(a.i,b.i)),
    _mm512_add_pd(_mm512_mul_pd(a.r,b.i),_mm512_mul_pd(a.i,b.r))};}
static inline Z scale(Z a,V b){return {_mm512_mul_pd(a.r,b),_mm512_mul_pd(a.i,b)};}
static inline Z div_real(Z a,V b){return {_mm512_div_pd(a.r,b),_mm512_div_pd(a.i,b)};}
static inline Z blend(__mmask8 m,Z a,Z b){return {_mm512_mask_blend_pd(m,a.r,b.r),_mm512_mask_blend_pd(m,a.i,b.i)};}
static inline void swap_mask(Z& a,Z& b,__mmask8 m){Z old=a;a=blend(m,a,b);b=blend(m,b,old);}
static inline Z divide(Z a,Z b){
    V exponent=_mm512_getexp_pd(_mm512_max_pd(_mm512_abs_pd(b.r),_mm512_abs_pd(b.i)));
    V negative=_mm512_sub_pd(number(0),exponent);
    V br=_mm512_scalef_pd(b.r,negative),bi=_mm512_scalef_pd(b.i,negative);
    V den=_mm512_add_pd(_mm512_mul_pd(br,br),_mm512_mul_pd(bi,bi));
    V re=_mm512_div_pd(_mm512_add_pd(_mm512_mul_pd(a.r,br),_mm512_mul_pd(a.i,bi)),den);
    V im=_mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(a.i,br),_mm512_mul_pd(a.r,bi)),den);
    return {_mm512_scalef_pd(re,negative),_mm512_scalef_pd(im,negative)};
}
static inline Z detpart(Z m[4][4],int a,int b,int c,int x,int y,int z){
    return mul(m[a][x],sub(mul(m[b][y],m[c][z]),mul(m[b][z],m[c][y])));
}
static inline Z cofactor(Z m[4][4],int i,int j){
    int a=(i+1)%4,b=(i+2)%4,c=(i+3)%4,x=(j+1)%4,y=(j+2)%4,z=(j+3)%4;
    return add(add(detpart(m,a,b,c,x,y,z),detpart(m,b,c,a,x,y,z)),detpart(m,c,a,b,x,y,z));
}
extern "C" int sg_table(int U,int C,int P,double noise,const int64_t* serving,
    const double* covariance,const double* signal,double* output){
    if(C!=3||U<=0||P<=0)return -1;
    int keys=U*P*4;
    for(int begin=0;begin<keys;begin+=8){
        alignas(64) double rr[4][4][8]{},ri[4][4][8]{},sr[4][4][8]{},si[4][4][8]{};
        for(int lane=0;lane<8;++lane){
            int key=std::min(begin+lane,keys-1),pair=key/4,mask=key%4,u=pair/P;
            int other[2],n=0;for(int c=0;c<C;++c)if(c!=serving[u])other[n++]=c;
            for(int i=0;i<4;++i)for(int j=0;j<4;++j){
                std::complex<double> r(i==j?noise:0.,0.);
                for(int bit=0;bit<2;++bit)if(mask&(1<<bit)){
                    int off=32*(pair*C+other[bit])+2*(i*4+j);r+=std::complex<double>(covariance[off],covariance[off+1]);
                }
                rr[i][j][lane]=r.real();ri[i][j][lane]=r.imag();
                int off=32*pair+2*(i*4+j);sr[i][j][lane]=signal[off];si[i][j][lane]=signal[off+1];
            }
        }
        Z r[4][4],s[4][4],x[4][4];__mmask8 permutations[4][4]{};
        for(int i=0;i<4;++i)for(int j=0;j<4;++j){
            r[i][j]={_mm512_load_pd(rr[i][j]),_mm512_load_pd(ri[i][j])};
            s[i][j]={_mm512_load_pd(sr[i][j]),_mm512_load_pd(si[i][j])};x[i][j]=s[i][j];
        }
        // Same pivoted lower LDLT recurrence as Eigen; independent lane pivots.
        for(int k=0;k<4;++k){
            alignas(64) double re[4][8],im[4][8];
            for(int i=k;i<4;++i){_mm512_store_pd(re[i],r[i][i].r);_mm512_store_pd(im[i],r[i][i].i);}
            for(int lane=0;lane<8;++lane){int p=k;double best=std::abs(std::complex<double>(re[k][lane],im[k][lane]));
                for(int i=k+1;i<4;++i){double v=std::abs(std::complex<double>(re[i][lane],im[i][lane]));if(v>best){p=i;best=v;}}
                permutations[k][p]|=__mmask8(1<<lane);
            }
            for(int p=k+1;p<4;++p){auto mask=permutations[k][p];
                for(int j=0;j<k;++j)swap_mask(r[k][j],r[p][j],mask);
                for(int i=p+1;i<4;++i)swap_mask(r[i][k],r[i][p],mask);
                swap_mask(r[k][k],r[p][p],mask);
                for(int i=k+1;i<p;++i){Z old=r[i][k];r[i][k]=blend(mask,r[i][k],conj(r[p][i]));r[p][i]=blend(mask,r[p][i],conj(old));}
                r[p][k]=blend(mask,r[p][k],conj(r[p][k]));
            }
            if(k){Z t[4];for(int j=0;j<k;++j)t[j]=scale(conj(r[k][j]),r[j][j].r);
                Z sum=mul(r[k][0],t[0]);for(int j=1;j<k;++j)sum=add(sum,mul(r[k][j],t[j]));
                r[k][k]=sub(r[k][k],sum);
                for(int i=k+1;i<4;++i){sum=mul(r[i][0],t[0]);for(int j=1;j<k;++j)sum=add(sum,mul(r[i][j],t[j]));r[i][k]=sub(r[i][k],sum);}
            }
            for(int i=k+1;i<4;++i)r[i][k]=div_real(r[i][k],r[k][k].r);
        }
        for(int k=0;k<4;++k)for(int p=k+1;p<4;++p)for(int j=0;j<4;++j)swap_mask(x[k][j],x[p][j],permutations[k][p]);
        for(int i=0;i<4;++i)for(int k=0;k<i;++k)for(int j=0;j<4;++j)x[i][j]=sub(x[i][j],mul(r[i][k],x[k][j]));
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)x[i][j]=div_real(x[i][j],r[i][i].r);
        for(int i=3;i>=0;--i)for(int k=3;k>i;--k)for(int j=0;j<4;++j)x[i][j]=sub(x[i][j],mul(conj(r[k][i]),x[k][j]));
        for(int k=3;k>=0;--k)for(int p=k+1;p<4;++p)for(int j=0;j<4;++j)swap_mask(x[k][j],x[p][j],permutations[k][p]);
        Z m[4][4];for(int i=0;i<4;++i)for(int j=0;j<4;++j){
            Z a=mul(conj(s[0][i]),x[0][j]),b=mul(conj(s[1][i]),x[1][j]);
            Z c=mul(conj(s[2][i]),x[2][j]),d=mul(conj(s[3][i]),x[3][j]);
            m[i][j]=add(add(a,b),add(c,d));if(i==j)m[i][j]=add({number(1),number(0)},m[i][j]);
        }
        Z row0[4];for(int j=0;j<4;++j){row0[j]=cofactor(m,j,0);if(j%2)row0[j]=neg(row0[j]);}
        Z det=add(add(mul(m[0][0],row0[0]),mul(m[1][0],row0[1])),add(mul(m[2][0],row0[2]),mul(m[3][0],row0[3])));
        for(int i=0;i<4;++i){Z e=divide(cofactor(m,i,i),det);V g=_mm512_max_pd(number(0),_mm512_sub_pd(_mm512_div_pd(number(1),e.r),number(1)));
            alignas(64) double result[8];_mm512_store_pd(result,g);
            for(int lane=0;lane<8&&begin+lane<keys;++lane)output[4*(begin+lane)+i]=result[lane];
        }
    }
    return 0;
}
