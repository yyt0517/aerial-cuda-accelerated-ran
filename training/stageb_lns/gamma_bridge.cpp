// Untouched original Eigen arithmetic, compiled with original architecture flags.
#include "physics_kernel.cpp"
extern "C" int sg_reference(int U,int C,int P,double noise,const int64_t* serving,
    const double* covariance,const double* signal,double* output){
    if(C!=3)return -1;
    for(int u=0;u<U;++u)for(int p=0;p<P;++p)for(int key=0;key<4;++key){
        Matrix r=Matrix::Identity()*noise;int bit=0;
        for(int c=0;c<C;++c)if(c!=serving[u]){
            if(key&(1<<bit)){Matrix v;int off=32*((u*P+p)*C+c);
                for(int j=0;j<16;++j)v.data()[j]=Complex(covariance[off+2*j],covariance[off+2*j+1]);r+=v;}
            ++bit;
        }
        Matrix s;int off=32*(u*P+p);for(int j=0;j<16;++j)s.data()[j]=Complex(signal[off+2*j],signal[off+2*j+1]);
        Matrix e=(Matrix::Identity()+s.adjoint()*r.ldlt().solve(s)).inverse();
        for(int l=0;l<4;++l)output[((u*P+p)*4+key)*4+l]=std::max(0.,1./e(l,l).real()-1.);
    }
    return 0;
}
