// Independent task-local B/2--exact AO--B/2 ring-polymer integrator.
// AO coefficients are supplied by the Python caller; no friction convention
// or thermal units are inferred here.  q/v are bead coordinates/derivatives.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
namespace {
using Complex=std::complex<double>;
constexpr double PI=3.141592653589793238462643383279502884;
constexpr double INV_SQRT2=0.707106781186547524400844362104849039;

struct Plan {
    int p; double scale;
    std::vector<int> reverse;
    std::vector<Complex> root;
    explicit Plan(int n):p(n),scale(1.0/std::sqrt(double(n))),reverse(n),root(n/2) {
        int bits=0; while((1<<bits)<p)++bits;
        for(int i=0;i<p;++i){int r=0,j=i;for(int b=0;b<bits;++b){r=(r<<1)|(j&1);j>>=1;}reverse[i]=r;}
        for(int i=0;i<p/2;++i){double angle=-2*PI*i/p;root[i]=Complex(std::cos(angle),std::sin(angle));}
    }
    void fft(Complex* a,bool inverse) const {
        for(int i=0;i<p;++i)if(reverse[i]>i)std::swap(a[i],a[reverse[i]]);
        for(int len=2;len<=p;len<<=1){int half=len/2,step=p/len;
            for(int start=0;start<p;start+=len)for(int j=0;j<half;++j){
                Complex w=inverse?std::conj(root[j*step]):root[j*step];
                Complex u=a[start+j],t=w*a[start+j+half];
                a[start+j]=u+t;a[start+j+half]=u-t;
            }
        }
        for(int i=0;i<p;++i)a[i]*=scale;
    }
};

// SplitMix64, one complete persistent state per trajectory.  Every call
// increments state before mixing. Box--Muller consumes complete pairs,
// retaining no hidden cached Gaussian across chunk boundaries.
inline uint64_t next_u64(uint64_t& s){
    uint64_t z=(s+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
inline void normals(uint64_t& state,double& a,double& b){
    double u=(double(next_u64(state)>>11)+0.5)*0x1.0p-53;
    double v=(double(next_u64(state)>>11)+0.5)*0x1.0p-53;
    double radius=std::sqrt(-2*std::log(u)),angle=2*PI*v;
    a=radius*std::cos(angle);b=radius*std::sin(angle);
}

bool valid(int n,int p,double dt,double wh){return n>0&&p>=2&&(p&(p-1))==0&&dt>0&&std::isfinite(dt)&&wh>0&&std::isfinite(wh);}
}

EXPORT const char* native_version(){return "rp1d-exact-AO-fft-public; uniform internal masses; splitmix64-boxmuller; orthogonal FFT";}

namespace {
// Independent 1D model: V(q)=q^2/2+alpha*q^4, physical mass=1.
// The quadratic harmonic and ring forces are entirely in the caller AO map.
void kick_1d(double* q,double* v,int p,double dt,double alpha,double internal_mass){
    if(alpha==0)return;
    double average_force=0;
    if(internal_mass!=1){for(int j=0;j<p;++j)average_force-=4*alpha*q[j]*q[j]*q[j];average_force/=p;}
    double inverse_mass=1/internal_mass,centroid_correction=(1-inverse_mass)*average_force;
    for(int j=0;j<p;++j)v[j]+=dt*(-4*alpha*q[j]*q[j]*q[j]*inverse_mass+centroid_correction);
}

void step_1d(const Plan& plan,const double* coeff,double* q,double* v,
             double dt,double alpha,double internal_mass,uint64_t& state,Complex* z,
             const double* explicit_noise=nullptr){
    int p=plan.p,m=p/2+1;
    kick_1d(q,v,p,dt/2,alpha,internal_mass);
    for(int j=0;j<p;++j)z[j]=Complex(q[j],v[j]);
    plan.fft(z,false);
    for(int k=0;k<m;++k){
        int opposite=(p-k)%p;
        Complex zk=z[k],zo=std::conj(z[opposite]);
        Complex Q=(zk+zo)*0.5,V=(zk-zo)*Complex(0,-0.5);
        const double* c=coeff+k*7;
        Complex newQ=c[0]*Q+c[1]*V,newV=c[2]*Q+c[3]*V;
        bool real=(k==0||k==p/2);
        if(c[4]!=0||c[5]!=0||c[6]!=0){
            double a,b,ai=0,bi=0;
            if(explicit_noise){const double* e=explicit_noise+k*4;a=e[0];b=e[1];ai=e[2];bi=e[3];}
            else {normals(state,a,b);if(!real)normals(state,ai,bi);}
            double scale=real?1:INV_SQRT2;
            newQ+=scale*Complex(c[4]*a,c[4]*ai);
            newV+=scale*Complex(c[5]*a+c[6]*b,c[5]*ai+c[6]*bi);
        }
        if(real){newQ=Complex(newQ.real(),0);newV=Complex(newV.real(),0);}
        z[k]=newQ+Complex(0,1)*newV;
        if(!real)z[opposite]=std::conj(newQ)+Complex(0,1)*std::conj(newV);
    }
    plan.fft(z,true);
    for(int j=0;j<p;++j){q[j]=z[j].real();v[j]=z[j].imag();}
    kick_1d(q,v,p,dt/2,alpha,internal_mass);
}

struct Moments1D {double energy,q2,q4,centroid,maxq,internal_variance,internal_force;bool finite;};
Moments1D observe_1d(const double* q,const double* v,int p,double alpha,double beta,double internal_mass){
    double energy=0,q2=0,q3=0,q4=0,centroid=0,vc=0,maxq=0,wp2=(p/beta)*(p/beta);bool finite=true;
    for(int j=0;j<p;++j){
        double x=q[j],xx=x*x,dx=x-q[(j+1)%p];
        energy+=.5*internal_mass*v[j]*v[j]+.5*xx+alpha*xx*xx+.5*wp2*dx*dx;
        q2+=xx;q3+=xx*x;q4+=xx*xx;centroid+=x;vc+=v[j];maxq=std::max(maxq,std::abs(x));
        finite=finite&&std::isfinite(x)&&std::isfinite(v[j]);
    }
    centroid/=p;vc/=p;energy=energy/p+.5*(1-internal_mass)*vc*vc;
    return {energy,q2/p,q4/p,centroid,maxq,q2/p-centroid*centroid,
            -4*alpha*(q3/p-centroid*centroid*centroid),finite&&std::isfinite(energy)};
}
}

// 1D API: q/v float64 (n,P), coeff float64 (P/2+1,7),
// persistent rng_state uint64 (n,), output float64 (n,1+nsteps/stride).
// diagnostics float64 (n,10): H0,Hend,max sampled relative drift,maxabsq,
// q2_initial,q2_final,q4_initial,q4_final,centroid_initial,centroid_final.
// obs nullable float64 (n,frames,2): within-ring variance,
// -4*alpha*(mean(q^3)-centroid^3).  internal_mass applies to ALL k>0;
// centroid mass stays one. coeff and initial velocities must match this mass.
EXPORT int native_run_1d(int n,int p,int nsteps,int stride,double dt,double alpha,
                         double beta,double internal_mass,const double* coeff,double* q,double* v,
                         uint64_t seed,uint64_t* rng_state,double* output,
                         double* diagnostics,double* obs,int threads){
    if(!valid(n,p,dt,1.)||nsteps<0||stride<1||nsteps%stride!=0||!(beta>0)||!(internal_mass>0)||!std::isfinite(internal_mass)||alpha<0||!std::isfinite(alpha))return -1;
    if(!coeff||!q||!v||!output)return -2;
    int m=p/2+1;for(int i=0;i<m*7;++i)if(!std::isfinite(coeff[i]))return -3;
    const Plan plan(p);int frames=1+nsteps/stride,bad=0;
    #pragma omp parallel num_threads(threads>0?threads:1) reduction(+:bad)
    {
        std::vector<Complex> buffer(p);
        #pragma omp for schedule(static)
        for(int i=0;i<n;++i){
            double* qi=q+size_t(i)*p;double* vi=v+size_t(i)*p;double* oi=output+size_t(i)*frames;
            uint64_t state=rng_state?rng_state[i]:seed+UINT64_C(0xd2b74407b1ce6e93)*uint64_t(i+1);
            Moments1D initial=observe_1d(qi,vi,p,alpha,beta,internal_mass),last=initial;
            double maxdrift=0,maxq=initial.maxq;oi[0]=initial.centroid;bool ok=initial.finite;int frame=1;
            double* ob=obs?obs+size_t(i)*frames*2:nullptr;
            if(ob){ob[0]=initial.internal_variance;ob[1]=initial.internal_force;}
            for(int j=1;j<=nsteps&&ok;++j){
                step_1d(plan,coeff,qi,vi,dt,alpha,internal_mass,state,buffer.data());
                if(j%stride==0){
                    last=observe_1d(qi,vi,p,alpha,beta,internal_mass);oi[frame]=last.centroid;
                    if(ob){ob[frame*2]=last.internal_variance;ob[frame*2+1]=last.internal_force;}++frame;
                    maxdrift=std::max(maxdrift,std::abs(last.energy-initial.energy)/std::max(1.0,std::abs(initial.energy)));
                    maxq=std::max(maxq,last.maxq);ok=last.finite;
                }
            }
            if(!ok){++bad;for(;frame<frames;++frame){oi[frame]=std::numeric_limits<double>::quiet_NaN();
                if(ob)ob[frame*2]=ob[frame*2+1]=std::numeric_limits<double>::quiet_NaN();}}
            if(rng_state)rng_state[i]=state;
            if(diagnostics){double* d=diagnostics+size_t(i)*10;
                d[0]=initial.energy;d[1]=last.energy;d[2]=maxdrift;d[3]=maxq;
                d[4]=initial.q2;d[5]=last.q2;d[6]=initial.q4;d[7]=last.q4;
                d[8]=initial.centroid;d[9]=last.centroid;
            }
        }
    }
    return bad;
}

// Same-noise 1D control; noise float64 (n,P/2+1,4) is standard normal.
EXPORT int native_step_explicit_1d(int n,int p,double dt,double alpha,
                                   double internal_mass,const double* coeff,const double* noise,
                                   double* q,double* v,int threads){
    if(!valid(n,p,dt,1.)||alpha<0||!std::isfinite(alpha)||!(internal_mass>0)||!std::isfinite(internal_mass)||!coeff||!noise||!q||!v)return -1;
    const Plan plan(p);int m=p/2+1;
    #pragma omp parallel num_threads(threads>0?threads:1)
    {
        std::vector<Complex> buffer(p);
        #pragma omp for schedule(static)
        for(int i=0;i<n;++i){uint64_t unused=0;step_1d(plan,coeff,q+size_t(i)*p,v+size_t(i)*p,
            dt,alpha,internal_mass,unused,buffer.data(),noise+size_t(i)*m*4);}
    }
    return 0;
}
