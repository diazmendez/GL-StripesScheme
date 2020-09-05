//***********************************************
// ldynamics.c: 
// Hace la dinamica de Langevin de un sistema con Hamitloniano
// de Ginzburg-Landau e interaccion dipolar. Es una adaptacion
// del programa del Lucas a un nuevo esquema numerico en el que
// uno pueda seguir con claridad los parametros que estan en la
// Tesis de Ale, identificar el campo critico, etc.
// Habana Vieja, 2009.
//***********************************************

//************************
// v1.0: calcula la J(k) y la B(k)
//************************

//************************
// v1.1: hace la dinamica e imprime fir a cada paso
// de tiempo
//************************

//************************
// v1.2: calcula las correlaciones temporales y las guarda 
// en ficheros
//************************

//************************
// Cambiamos los parametros por otros mas comodos y redefinimos
// la transformada. 
// Cambiamos la definicion de B(k).
//************************

//************************
// v1.3: deadend.
//************************

//************************
// v1.4: crea un subdirectorio con su read.me que contiene la data
// de las correlaciones corridas y una carpeta con las fotos para
// cada tw.
//************************


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./fftw3.h"
//#include <limits.h> 

// Compile: gcc -O3 -Wall -o ldy.out ldynamics.c -lfftw3 -L ./includes/pg46/lib/ -lm


#define pi acos(-1)
#define SIGN1(b) ((b) >= 0.0 ? 1.0 : -1.0)

#define alfa (1.0/25.0)
#define beta (325.0)
#define delta ((1.0-pi*alfa)/2.0)



#define hc 0.0 // esto es solo para el readme
#define l (m*n)
#define N (l*l)
#define a (1.0/m)

#define varianza sqrt(2*T*dt) // Esto hay que revisitarlo!
#define triangle 0.1


#define points 20 


#define ntw 15


//*******************************************
//******* Generador de aleatorios
#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)         

unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned rand4init(void)
{
    unsigned long long y;
    
    y = (myrand*16807LL);
    myrand = (y&0x7fffffff) + (y>>31);
    if (myrand&0x8000000)
	myrand = (myrand&0x7fffffff) + 1;
    return myrand;
}                    

void Init_Random(void)
{
    int i;
    
    ip = 128;
    ip1 = ip - 24;
    ip2 = ip - 55;
    ip3 = ip - 61;
    
    for (i=ip3; i<ip; i++)
	ira[i] = rand4init();
}                    

float gauss_ran(void)
{
    static int iset=0;
    static float gset;
    float fac, rsq, v1, v2;
    
    if (iset == 0) {
	do {
	    v1 = 2.0 * FRANDOM - 1.0;
	    v2 = 2.0 * FRANDOM - 1.0;
	    rsq = v1 * v1 + v2 * v2;
	} while (rsq >= 1.0 || rsq == 0.0);
	fac = sqrt(-2.0 * log(rsq) / rsq);
	gset = v1 * fac;
	iset = 1;
	return v2 * fac;
    } else {
	iset = 0;
	return gset;
    }
}
//*******************************************
//*******************************************
 



// global variables
int m, n, tmax;
double T, H;
char *dir, *via;
double *fir, *fi3r, *Bk, *localr, *BuJ;
fftw_complex *Jk, *fik, *fi3k, *localk;
fftw_plan f3r2c,f3c2r,fr2c,fc2r,localc2r;
double *fitw;

int *tw, ns, samples; 
double *Cinit;

int ctw, t;

char pfile[90],rfile[90],datafile[90];

unsigned myrand_back;
int *right,*left,*up,*down;

double dt;
int ptwinit;


// functions
void set_memory();
void erase_memory();
void init_Bk();
void init_fir();
void init_Jk();
void print_config(int);
void redef_B_J();
void tstep();
void do_correlations(int);
void get_fi();
void init_tw();
void make_files();
double set_energy();
void init_BuJ();
double orientacional();
void neighbours_2d();
void write_data();


//************************* Main program
int main(int argc, char *argv[])
{
  int  tnext, num;
  
  if(argc!=12) {
    printf("usage: %s <m> <n> <Jkfile> <T> <H> <dt> <tmax> <ptwinit> <samples> <directorio> <seed>\n", argv[0]);
    exit(1);
  }
  
  
  m=(int)atoi(argv[1]);
  n=(int)atoi(argv[2]);
  via = (char *)argv[3];
  T=(double)atof(argv[4]);
  H=(double)atof(argv[5]);
  dt=(double)atof(argv[6]);
  tmax=(int)atoi(argv[7]);
  ptwinit=(int)atoi(argv[8]);
  samples=(int)atoi(argv[9]);
  dir=(char *)argv[10];
  myrand=(unsigned)atoi(argv[11]); 

  myrand_back=myrand;
  

  Init_Random();

//printf("voy a dar memoria...\n");
  set_memory();

//printf("voy a hacer los vecinos...\n");
  neighbours_2d();

//printf("voy a hacer el Bk...\n");    
  init_Bk();

//printf("voy a hacer el Jk...\n");
  init_Jk();


//printf("voy a hacer los tw...\n");
  init_tw();

//printf("voy a hacer el BuJ...\n");
//  init_BuJ();

//printf("voy a hacer el redefine...\n");
  redef_B_J();
  


//printf("voy a crear los ficheros...\n");
  
  make_files();
  
  for (ns=0;ns<samples;ns++)
    {  
//printf("voy a asignar condiciones iniciales\n");
      init_fir();
      t=0;num=0;ctw=0;
      
      //printf("%i\t%f\t%f\n",t,set_energy(),orientacional());
//printf("voy a escribir la data\n");
      write_data();
      //print_config(1);
//printf("lo hice\n");      
      do
	{
	  tnext=t+(int)pow(10,(double)num/points);  // la escala logaritmica.
	  tnext=(tnext>tmax?tmax+1:tnext); //esto es para no pasar ni un paso mas que tmax.
	  if (tnext>t)
	    {
	      if ((tnext<tw[ctw])||(t>=tw[2*ntw-1])) //si la proxima parada es tstep.
		{ 
		  for (;t<tnext;t++) tstep();
		}
	      else 
		{
		  for (;t<tw[ctw];t++) tstep();
		  get_fi(); //guarda la configuracion en fitw y actualiza Cinit.
		  ctw+=(ctw<2*ntw-1?1:0); // si ctw es el mayor no aumenta mas.
		  num=0; // reinicia la escala.
		  print_config(1);
		}
	      //printf("%i\t%f\t%f\n",t,set_energy(),orientacional());
	      write_data();
	      do_correlations(0); // el entero es para imprimir la o no las fotos.
	    }
	  num++;
	}while (t<tmax);
      //print_config(1);
    }
  
  
  

  erase_memory();
  
  return 0;
}







//**** da memoria a los arreglos y define los planes
void set_memory()
{
  fir = fftw_malloc(sizeof(double)*N);
  fik = fftw_malloc(sizeof(fftw_complex)*(l*(l/2+1)));
  fi3r = fftw_malloc(sizeof(double)*N);
  fi3k = fftw_malloc(sizeof(fftw_complex)*(l*(l/2+1)));

  Jk = fftw_malloc(sizeof(fftw_complex)*(l*(l/2+1)));
  Bk = malloc(sizeof(double)*(l*(l/2+1)));

  localr = fftw_malloc(sizeof(double)*N);
  localk = fftw_malloc(sizeof(fftw_complex)*(l*(l/2+1)));
  BuJ = malloc(sizeof(double)*(l*(l/2+1)));
    
  fr2c = fftw_plan_dft_r2c_2d(l,l,fir,fik,FFTW_PATIENT);
  fc2r = fftw_plan_dft_c2r_2d(l,l,fik,fir,FFTW_PATIENT);
  f3r2c = fftw_plan_dft_r2c_2d(l,l,fi3r,fi3k,FFTW_PATIENT);
  localc2r = fftw_plan_dft_c2r_2d(l,l,localk,localr,FFTW_PATIENT);


  fitw = (double *) malloc(2*ntw*N*sizeof(double));
  tw = (int *) malloc(2*ntw*sizeof(int));
  Cinit = (double *) malloc(2*ntw*sizeof(double));

  right=(int *) malloc(N*sizeof(int)); 
  left=(int *) malloc(N*sizeof(int)); 
  up=(int *) malloc(N*sizeof(int)); 
  down=(int *) malloc(N*sizeof(int)); 

}



//****** libera la memoria de arreglos y planes
void erase_memory()
{
  fftw_destroy_plan(fr2c);
  fftw_destroy_plan(fc2r);
  fftw_destroy_plan(f3r2c);
  fftw_destroy_plan(localc2r);

  fftw_free(fir);  fftw_free(fik);  
  fftw_free(fi3r);  fftw_free(fi3k);

  fftw_free(localr);  fftw_free(localk);

  fftw_free(Jk);
  free(Bk);
  free(BuJ);

  free(fitw);
  free(Cinit);
  free(tw);

  free(right); free(left); free(up); free(down);
  return;
}




//****** define el arreglo de los B(k)
void init_Bk()
{
  int i, j, cnt=0;


  for (i=0; i<l; i++)
    for (j=0;j<l/2+1;j++)
      {
	if(i<l/2 && j<l/2)
	Bk[cnt]=pow(2*pi*i/l,2)+pow(2*pi*j/l,2);
        if(i>=l/2 && j<l/2)
	Bk[cnt]=pow(2*pi*i/l-2*pi,2)+pow(2*pi*j/l,2);
        if(i<l/2 && j==l/2)
	Bk[cnt]=pow(2*pi*i/l,2)+pow(2*pi*j/l-2*pi,2);
        if(i>=l/2 && j==l/2)
	Bk[cnt]=pow(2*pi*i/l-2*pi,2)+pow(2*pi*j/l-2*pi,2);
	
        //Bk[cnt]=8-pow(2*pi*i/l-pi,2)-pow(2*pi*j/l-pi,2); //2*(2-cos(2*pi*i/l)-cos(2*pi*j/l))
	cnt++;
      }
}



//****** inicializa el campo
void init_fir()
{
  int i;

  // la funcion sigma*gauss_ran() redefine la sigma de la gaussiana 
  for (i=0; i<N; i++) fir[i]=sqrt(triangle)*gauss_ran(); 

}




//****** hace las sumas de ewald e inicializa Jk
void init_Jk()
{

  int i, j, cnt=0;
  FILE *opfd;
  float temp;

  opfd = fopen(via, "r");

  for (i=0;i<l;i++)
    {
      for(j=0;j<l/2+1;j++)
        {
          fscanf(opfd,"%f",&temp);
          Jk[cnt][0]=temp;
          Jk[cnt][1]=0;
          //printf("%f-%f  ",temp,Jk[cnt][0]);
          cnt++;
        }
      fscanf(opfd,"\n");
      //printf("\n");
    }
  fclose(opfd);

  return;
}






void redef_B_J()
{
  int i;

  for (i=0;i<l*(l/2+1);i++) 
    Jk[i][0]=(1-(a/delta)*Jk[i][0]*dt)/(1+dt*Bk[i]);

  for (i=0;i<l*(l/2+1);i++) 
    Bk[i]=1.0/(1+dt*Bk[i]);


}


void print_config(int pp)
{
  int i,j;
  FILE *opfp;

  int lf=300; // siempre lf<=l

  if (pp==1)
    {
      sprintf(pfile,"%s/fotos/ph_t_%i_ns_%i.dat",dir,t,ns);
      opfp = fopen(pfile, "a");
      
      for (i=0;i<lf;i++)
	for (j=0;j<lf;j++)
	  fprintf(opfp,"%f\t%f\t%f\n",(double)i,(double)j,fir[i*l+j]);
      fprintf(opfp,"\n");
      fflush(opfp);
      
      fclose(opfp);
    }
  else
    {
      for (i=0;i<l;i++)
	for (j=0;j<l;j++)
	  printf("%f\t%f\t%f\n",(double)i,(double)j,fir[i*l+j]);
      printf("\n");
      fflush(0);
      
    }
      
}


void tstep()
{
  int i;

  
  for (i=0;i<N;i++)
    fi3r[i]=(fir[i]-fir[i]*fir[i]*fir[i])*dt*beta*a*a + varianza*gauss_ran() + H*dt*a*a;

  // transformando
  fftw_execute(fr2c);
  fftw_execute(f3r2c);


  for (i=0;i<l*(l/2+1);i++) 
    {     
      fik[i][0]=Jk[i][0]*fik[i][0] + Bk[i]*fi3k[i][0];
      fik[i][1]=Jk[i][0]*fik[i][1] + Bk[i]*fi3k[i][1];
    }  

  // antitransformando y normalizando
  fftw_execute(fc2r);
  for (i=0;i<N;i++) fir[i]=fir[i]/N;

}







//************************************************************************



void get_fi()
{

    int i;    
    
    Cinit[ctw]=0;

    for (i=0;i<N;i++) 
      {
	fitw[(ctw*N)+i]=fir[i]; 
	Cinit[ctw]+=fir[i]*fir[i];
      }
    Cinit[ctw]=Cinit[ctw]/N;
    //for (i=0;i<l;i++)
    //for (j=0;j<l;j++)
    //printf("%f\t%f\t%f\n",(double)i,(double)j,fitw[i*l+j]);     
    
}

void do_correlations(int pp)
{
  int i,atw;
  double Co;
  FILE *opfco;
  char cofile[60];
  
  for (atw=0;atw<ctw;atw++)
    {
      Co=0;
      for (i=0;i<N;i++) 
	Co+=(fir[i]*fitw[(atw*N)+i]);

      if (div(atw,2).rem==0)
	sprintf(cofile,"%s/corr_tw2_%i_ns_%i.dat",dir,ptwinit+atw/2,ns);
      else
	sprintf(cofile,"%s/corr_tw2_%i.5_ns_%i.dat",dir,ptwinit+div(atw,2).quot,ns);
	
      opfco=fopen(cofile, "a");
      fprintf(opfco,"%i\t%f\t%3.15f\n",t-tw[atw],Co/N/Cinit[atw],Co/N);
      fflush(opfco);
      fclose(opfco);
      //printf("pasé\n\n");
    }  
  
  if (pp==1) print_config(1);
  
}


void init_tw()
{
  int i;

  // lucas tenia:
  //tw={1,16,64,128,256,512,1024,2048,4096,8192,16384,32768,47000,65536,82000};

  for (i=ptwinit;i<ptwinit+ntw;i++) 
    {
      tw[2*(i-ptwinit)]=pow(2,i);
      tw[2*(i-ptwinit)+1]=pow(2,i+.5);
    }

}


//************************************
void make_files()
{
    FILE *opf;
    char order[60];

    sprintf(rfile,"%s/README.dat",dir);    
    
    sprintf(order,"mkdir %s",dir);
    system(order); 
    sprintf(order,"mkdir %s/fotos",dir);
    system(order);
    opf = fopen(rfile, "w");
    fprintf(opf,"***********************************************************************************************\n");
    fprintf(opf,"Aclaracion del funcionamiento del programa:\n");
    fprintf(opf,"Comienza con una cofiguracion desordenada y se hace evolucionar (quench) a T y H\n");
    fprintf(opf,"un numero tmax de pasos montecarlo. Se mide por un numero no menor que points por\n");
    fprintf(opf,"decada las correlaciones y la configuracion del sistema\n");

    fprintf(opf,"\n");
    fprintf(opf,"Asi se generan en el subdirectorio creado <directorio> el subdirectoriofotos. \n");
    fprintf(opf,"\n");
    fprintf(opf,"fotos: tiene los ficheros ph_t_?_ns_?.dat que son fotos  del sistema en cada valor de t  para cada sample."); 
    fprintf(opf,"correspondientes al sample X, es decir,  un juego \n");
    fprintf(opf,"Estas fotos son un juego de de 3 columnas y L*L filas y corresponden a i,j,fir(i,j)\n");
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fprintf(opf,"En el subdirectorio estaran los ficheros corr_tw2_?_ns_?.dat, donde el primer numero es la potencia de 2 que corresponde al tw y el segundo es el numero del sample. \n");
    fprintf(opf,"corr_tw2_?_ns_?.dat: tiene un juegos de datos de tres columnas, la primera es el valor de t-tw, la segunda la correlacion normalizada y la tercera la correlacion sin normalizar.\n");
    fprintf(opf,"\n");
    fprintf(opf,"Se crea ademas un fichero llamado data_ns_?.dat por cada sample, que contiene tres columnas que\n");
    fprintf(opf,"son: tiempo, energia y parametro de orden orientacional.\n");
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fprintf(opf,"Los parámetros explicitos de entrada de esta corrida son:\n");
    fprintf(opf,"\n");
    fprintf(opf,"       L = %i\n", l);
    fprintf(opf,"       T = %f\n", T);
    fprintf(opf,"       H = %f\n", H);
    fprintf(opf,"      dt = %f\n", dt);
    fprintf(opf,"    tmax = %i\n", tmax);
    fprintf(opf," ptwinit = %i\n", ptwinit);
    fprintf(opf," samples = %i\n", samples);
    fprintf(opf,"    seed = %u\n", myrand_back);
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fprintf(opf,"Los parámetros implicitos en esta corrida son:\n");
    fprintf(opf,"\n");
    fprintf(opf,"     alfa = %f\n", alfa);
    fprintf(opf,"     beta = %f\n", beta);
    //    fprintf(opf,"    gamma = %f\n", gamma);
    fprintf(opf,"       hc = %f\n", hc);
    fprintf(opf," varianza = %f\n", varianza);
    fprintf(opf," triangle = %f\n", triangle);
    fprintf(opf,"   points = %i\n", points);
    fprintf(opf,"      ntw = %i\n", ntw);
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fclose(opf);
}



void init_BuJ() 
{
  int i;

  for(i=0;i<l*(l/2+1);i++)
      BuJ[i] = alfa*Bk[i]+Jk[i][0];

}


double orientacional()
{
  int i;
  double s[N],nx,ny,qtos=0.0,q11=0.0,q12=0.0,DELTA;

  for(i=0;i<N;i++)
    s[i]=SIGN1(fir[i]);

  for(i=0;i<N;i++){
    DELTA=fabs(s[down[i]]-s[up[i]])/2. + fabs(s[left[i]]-s[right[i]])/2. -
      fabs(s[down[i]]-s[up[i]])*fabs(s[left[i]]-s[right[i]])/4.;

    qtos+=DELTA;

    ny = fir[left[i]]-fir[right[i]];
    nx = fir[down[i]]-fir[up[i]];
    q11+= (2.0*nx*nx/(nx*nx+ny*ny)-1.0)*DELTA;
    q12+= (2.0*nx*ny/(nx*nx+ny*ny))*DELTA;
  }

  return sqrt(q11*q11+q12*q12)/qtos;
}



/*****************************************************************
***                          2D Neighbours                     ***
***                    Last Modified: 24/01/1999               ***
***                                                            ***
***  In a 2D, with periodic boundary conditions, create the    ***
***  matrixes with the neighbours of all sites.                ***
***                                                            ***
***  Input: right,left,up & down -> matrixes with neighb. sites***
***         l_size -> linear dimension (l_size x l_size)       ***
***                                                            ***
**   From right to left, top to bottom, front to back.         ***
*****************************************************************/
void neighbours_2d()
{
  int i,l2,lsize;
  lsize=l;
  l2 = lsize*lsize;
  for (i=0; i<l2; ++i)     
    {
      if (i % lsize==lsize-1) *(right+i) = i-lsize+1;/* last col. */
      else *(right+i) = i+1; 
      if (i % lsize==0) *(left+i) = i+lsize-1;       /* first col.*/
      else *(left+i) = i-1;
      if (i<lsize) *(up+i) = l2-lsize+i;             /* first row */
      else *(up+i) = i - lsize;
      if (i>=l2-lsize) *(down+i) = (i % lsize);      /* last row  */
      else *(down+i) = i + lsize;
    }  
  return;
}


void write_data()
{

  FILE *opfd;
  int jj;
  double fim=0, fi2m=0;


  for (jj=0;jj<N;jj++){fim+=fir[jj];fi2m+=fir[jj]*fir[jj];}


  sprintf(datafile,"%s/data_ns_%i.dat",dir,ns);
  opfd = fopen(datafile, "a");
  //fprintf(opfd,"%i\t%f\t%f\n",t,set_energy(),orientacional());
  fprintf(opfd,"%i\t%f\t%f\n",t,fim/N,fi2m/N);
  //fprintf(opfd,"#si\n");
  fflush(opfd);
  fclose(opfd);

}



double set_energy() //esta energia hay que cambiarla!
{
  double fac=1.0/N,e=0.0;
  int i;


  fftw_execute(fr2c);
  
  /*calcula campo local (laplaciano + dipolar)*/
  for(i=0;i<l*(l/2+1);i++){
    localk[i][0] = fik[i][0]*BuJ[i];
    localk[i][1] = fik[i][1]*BuJ[i];
  }
  
  fftw_execute(localc2r);
  
  for(i=0;i<N;i++)
      e +=beta*(-pow(fir[i],2)+.5*pow(fir[i],4))-2.0*H*fir[i]
	+fac*fir[i]*localr[i];




  return (e/2.0/N);
}
