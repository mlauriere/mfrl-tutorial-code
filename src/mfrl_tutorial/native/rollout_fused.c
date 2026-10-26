/* Fused full-population rollout. No fast-math, no contraction, double precision.
 * Inputs retain NumPy's exact agent/time/sample/coordinate RNG ordering.
 * This is a local ctypes library, not an installed Python extension. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int rollout_population(size_t M,size_t N,size_t T,double gamma,
 const double *fk,const double *fl,const double *sk,const double *sl,
 const double *p0,const double *c0,const double *pn,const double *cn,double *out){
 double *x0=malloc(N*M*sizeof(double)),*x1=malloc(N*M*sizeof(double));
 double *z0=malloc(M*sizeof(double)),*z1=malloc(M*sizeof(double)),*cost=malloc(M*sizeof(double));
 if(!x0||!x1||!z0||!z1||!cost){free(x0);free(x1);free(z0);free(z1);free(cost);return 1;}
 for(size_t n=0;n<N;n++) for(size_t m=0;m<M;m++){
  size_t a=n*M+m;x0[a]=p0[2*a]+c0[2*m];x1[a]=p0[2*a+1]+c0[2*m+1];
 }
 memset(out,0,M*sizeof(double));
 for(size_t t=0;t<T;t++){
  memset(z0,0,M*sizeof(double));memset(z1,0,M*sizeof(double));memset(cost,0,M*sizeof(double));
  for(size_t n=0;n<N;n++) for(size_t m=0;m<M;m++){z0[m]+=x0[n*M+m];z1[m]+=x1[n*M+m];}
  for(size_t m=0;m<M;m++){z0[m]/=N;z1[m]/=N;}
  for(size_t n=0;n<N;n++) for(size_t m=0;m<M;m++){
   const size_t a=n*M+m,b=4*m,j=((n*T+t)*M+m)*2;
   const double y0=x0[a]-z0[m],y1=x1[a]-z1[m];
   cost[m]+=sk[b]*y0*y0+(sk[b+1]+sk[b+2])*y0*y1+sk[b+3]*y1*y1;
   /* Match reference addition association. Both old coordinates are held. */
   x0[a]=fk[b]*y0+fk[b+1]*y1+fl[b]*z0[m]+fl[b+1]*z1[m]+pn[j]+cn[(t*M+m)*2];
   x1[a]=fk[b+2]*y0+fk[b+3]*y1+fl[b+2]*z0[m]+fl[b+3]*z1[m]+pn[j+1]+cn[(t*M+m)*2+1];
  }
  double discount=pow(gamma,(double)t);
  for(size_t m=0;m<M;m++){size_t b=4*m;out[m]+=discount*(cost[m]/N+sl[b]*z0[m]*z0[m]+(sl[b+1]+sl[b+2])*z0[m]*z1[m]+sl[b+3]*z1[m]*z1[m]);}
 }
 free(x0);free(x1);free(z0);free(z1);free(cost);return 0;
}
