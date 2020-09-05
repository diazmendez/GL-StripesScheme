//************************************************
// stldynamics.c:
// Hace simulated annealing bajando en temperatura un sistema de
// Ginzburg-Landau con interaccion dipolar y campo aplicado, por
// medio de una dinamica de Langevin. 
// La Habana, 2009 - Porto Alegre, ????  
//************************************************

// La idea es definir una temperatura inicial Tmax (alta), una final Tmin (baja) y un numero de puntos points. Con esto se va bajando la temperatura desde Tmax hasta Tmin por con un decrecimiento de dT=(Tmax-Tmin)/points. En cada valor de temperatura se deja el sistema relajar un numero relax de pasos y luego, durante otro cantidad relax de pasos se miden medh valores de las magnitudes de interes (energia, magnetizacion total, parametro orientacional...), que se guardan en ficheros, un fichero para cada magnitud. La estructura de cada uno de estos ficheros es de points filas y medh+1 columnas, cada fila corresponde a un valor de T, que sera el primer elemento de la fila, mientras que los medh elementos restantes van a ser diferentes valores de la magnitud. Asi la data servira tanto para calcular la magnitud media, cuadrada, etc. como tambien para hallar la distribucion de esa magnitud en cada valor de T.
// Tambien se guarda, para cada T un numero ph de fotos del sistema, con el animo de ver despues las configuraciones tipicas del sistema, calcular correlaciones, factores de estructura, etc.

//Compile: gcc -O3 -Wall -o stldy.out stldynamics.c -lfftw3 -L ./includes/pg46/lib/ -lm
//Execution example: ./stldy.out 100 0.001 0.01 0.0 60 100000 1000 50 corrida1 483957


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./fftw3.h"


#define pi acos(-1)
#define SIGN1(b) ((b) >= 0.0 ? 1.0 : -1.0)

#define alfa (1.0/5.0)
#define beta (120.0)
#define delta ((1.0-pi*alfa)/2.0)



#define l (mm*nn)
#define N (l*l)
#define a (1.0/mm)

#define varianza sqrt(2*T*dt)
#define triangle 1.0

#define dt 0.15 //valor real de tiempo en cada paso.

//********************************************************************
//******* Esto es solo para el Generador de numeros aleatorios
//********************************************************************
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
//****************************************************************
//************* Fin del Generador de numeros aleatorios **********
//****************************************************************


// global variables
//int l; //tamanho del sistema.
double T, Tmax, Tmin, dT; //temperatura, maxima, minima y decremento.
double H;//campo.
char *dir;//directorio que crea para meter los datos.
double *Jr, *fir, *fi3r, *Bk, *localr, *BuJ;  //punteros para hacer el paso de... 
fftw_complex *Jk, *fik, *fi3k, *localk;       //...langevin, mira el programa de la... 
fftw_plan f3r2c,f3c2r,fr2c,fc2r,localc2r;     //...dinamica, alli esta mas claro.
char peafile[60],pefile[60],efile[60],pfile[60],rfile[60],mfile[60],phfile[60]; //guardan el nombre de los fichers de perimetro average, perimetro, energia, parametro, readme, magnetizacion y fotos, respectivamente.
unsigned myrand_back;//esto es solo para recordar la semilla en al readme.
int *right,*left,*up,*down;//arreglos de vecinos.
int points, relax, medh, ph;//numero de puntos de T, tiempo de relajacion, numero de mediciones para el histograma en cada T, numero de fotos en cada T.
char *via;
int mm,nn;




//functions
void set_memory(); 
void erase_memory();
void neighbours_2d();
void init_Bk();
void init_fir();
void init_Jk();
void print_config(int);
void redef_B_J();
void tstep();
void make_files();
double set_energy();
void init_BuJ();
double orientacional();
double set_magnet();
void write_data(int);
double set_perimeter(int);
int opposite_n(int,int);



//*******************************************
int main(int argc, char *argv[])
{
  int i, j;
  int oave, steph, stepph; //tiempo total durante el cual se "mide", tiempo entre una medicion y otra, cantidad de mediciones entre una foto y otra.

  if(argc!=13) // espera por 10 datos en la linea de ejecucion del programa.
    {
      printf("usage: %s <m> <n> <JkFile> <H> <Tmax> <Tmin> <points> <relax> <medh(histogram)> <ph(fotos))> <dir> <seed>\n", argv[0]);
      exit(1);
    }
  
  
  mm=(int)atoi(argv[1]); //   
  nn=(int)atoi(argv[2]); //   
  via=(char *)argv[3]; //   
  H=(double)atof(argv[4]);//campo.
  Tmax=(double)atof(argv[5]);//temperatura maxima.
  Tmin=(double)atof(argv[6]);//temperatura minima.
  points = (int)atoi(argv[7]);//numero de puntos.
  relax = (int)atoi(argv[8]);//tiempo de relajacion.
  medh = (int)atoi(argv[9]);//cantidad de mediones en cada T.
  ph = (int)atoi(argv[10]);//cantidad de fotos en cada T.
  dir = (char *) argv[11];//nombre del subdirectorio de datos.
  myrand = (unsigned)atoi(argv[12]);//semilla del generador de aleatorios.

  myrand_back=myrand; //guarda la semilla del generador de aleatorios.
 
  Init_Random(); // inicializa el  generador de aleatorios.
  
  set_memory(); // da memoria a los arreglos y define los planes de fftw.
  
  neighbours_2d(); // inicializa los vecinos.
    
  init_Bk(); // inicializa Bk.
  init_Jk(); // inicializa Jk.

  

  init_BuJ(); // inicializa el arreglo BuJ (ver el programa de dinamica).
  redef_B_J(); // redefine Bk y Jk (ver el programa de dinamic).
  
  make_files(); // crea el subdirectorio, hace el readme y da nombre a los archivos (excepto los de fotos).

  oave=relax; // define que se medira durante un tiempo igual a relax.
  
  steph=(int)(oave/(medh+1)); // calcula el tiempo entre mediciones de histograma.
  stepph=(int)(medh/(ph+1)); // calcula la cantidad de mediciones de histograma entre una foto y otra.

  init_fir(); // inicializa a fi como un ruido gausiano. 
  fftw_execute(fr2c); // transforma a fi.

  dT=(double)(Tmax-Tmin)/points-0.0000000000001; // calcula dT.

 
 
  for (T=Tmax;T>=Tmin;T-=dT) // ciclo principal, T disminuye en dT en cada vuelta.
    {
      for (i=0;i<relax;i++) tstep(); // relaja relax pasos de dinamica.
      write_data(0); // escribe la temperatura en cada fichero de magnitudes.      
      
      for(i=0;i<medh;i++) // ciclo por el numero de mediciones.
	{
	  for (j=0;j<steph;j++) tstep(); // relaja steph pasos antes de medir.
	  write_data(1); // escribe los datos en los ficheros de magnitudes.
	  if(div(i,stepph).rem==0) print_config(1); // guarda foto del sistema si es hora de guardarla.
	}
      write_data(2); //escribe un salto de linea en los ficheros de magnitudes.      
    }

  
  erase_memory(); // libera la memoris de los punteros y los planes.

  return 1;
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


  free(right); free(left); free(up); free(down);
  return;
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



//************************************
void make_files()
{
    FILE *opf;
    char order[30];

    // dando nombres a los ficheros
    sprintf(efile,"%s/energy.dat",dir);
    sprintf(pfile,"%s/param_o.dat",dir);
    sprintf(mfile,"%s/magnet.dat",dir);
    sprintf(pefile,"%s/perim.dat",dir);
    sprintf(peafile,"%s/perim_a.dat",dir);
    sprintf(rfile,"%s/README.dat",dir);    

    // creando subdirectorios    
    sprintf(order,"mkdir %s",dir);
    system(order); 
    sprintf(order,"mkdir %s/fotos",dir);
    system(order);

    // escribiendo el README.dat
    opf = fopen(rfile, "w");
    fprintf(opf,"\n");
    fprintf(opf,"***********************************************************************************************\n");
    fprintf(opf,"\n");
    fprintf(opf,"La idea es definir una temperatura inicial Tmax (alta), una final Tmin (baja) y un numero de puntos points. Con esto se va bajando la temperatura desde Tmax hasta Tmin por con un decrecimiento de dT=(Tmax-Tmin)/points. En cada valor de temperatura se deja el sistema relajar un numero relax de pasos y luego, durante otro cantidad relax de pasos se miden medh valores de las magnitudes de interes (energia, magnetizacion total, parametro orientacional...), que se guardan en ficheros, un fichero para cada magnitud. La estructura de cada uno de estos ficheros es de points filas y medh+1 columnas, cada fila corresponde a un valor de T, que sera el primer elemento de la fila, mientras que los medh elementos restantes van a ser diferentes valores de la magnitud. Asi la data servira tanto para calcular la magnitud media, cuadrada, etc. como tambien para hallar la distribucion de esa magnitud en cada valor de T.\n");
    fprintf(opf,"Tambien se guarda, para cada T un numero ph de fotos del sistema, con el animo de ver despues las configuraciones tipicas del sistema, calcular correlaciones, factores de estructura, etc.\n");
    fprintf(opf,"\n");
    fprintf(opf,"Los parámetros explicitos de entrada de esta corrida son:\n");
    fprintf(opf,"\n");
    fprintf(opf,"       m = %i\n", mm);
    fprintf(opf,"       n = %i\n", nn);
    fprintf(opf,"       H = %f\n", H); 
    fprintf(opf,"    Tmax = %f\n", Tmax);
    fprintf(opf,"    Tmin = %f\n", Tmin);
    fprintf(opf,"  points = %i\n", points);
    fprintf(opf,"   relax = %i\n", relax);
    fprintf(opf,"    medh = %i\n", medh);
    fprintf(opf,"      ph = %i\n", ph);
    fprintf(opf,"    seed = %u\n", myrand_back);
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fprintf(opf,"Los parámetros implicitos en esta corrida son:\n");
    fprintf(opf,"\n");
    fprintf(opf,"     alfa = %f\n", alfa);
    fprintf(opf,"     beta = %f\n", beta);
    //fprintf(opf,"    gamma = %f\n", gamma);
    fprintf(opf," triangle = %f\n", triangle);
    fprintf(opf,"\n");
    fprintf(opf,"\n");
    fclose(opf);
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
      e +=beta*(-pow(fir[i],2)+.5*pow(fir[i],4))-  2.0*H*fir[i]
	+fac*fir[i]*localr[i];




  return (e/2.0/N);
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


void print_config(int pp)
{
  int i,j;
  FILE *fph;

  
  if (pp==1)
    {
      sprintf(phfile,"%s/fotos/phT%f.dat",dir,T); // dando nombre al archivo de fotos
      fph = fopen(phfile, "a");
      for (i=0;i<l;i++)
	for (j=0;j<l;j++)
	  fprintf(fph,"%f\t%f\t%f\n",(double)i,(double)j,fir[i*l+j]);
      fprintf(fph,"\n");
      fclose(fph);
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

double set_magnet()
{
  int i;
  double m=0.0;
  for (i=0;i<N;i++) m+=fir[i];

  return fabs(m/N);
}

void write_data(int data)
{

  FILE *fen, *fpao, *fma, *fpe, *fpea;

  
  if (data==0)//escribe solo el valor de temperatura.
    {      
      fen = fopen(efile, "a");
      fpao = fopen(pfile, "a");
      fma = fopen(mfile, "a");
      fpe = fopen(pefile, "a");
      fpea = fopen(peafile, "a");
      fprintf(fen, "%f\t", T);
      fprintf(fpao, "%f\t", T);
      fprintf(fma, "%f\t", T);
      fprintf(fpe, "%f\t", T);
      fprintf(fpea, "%f\t", T);
      fclose(fen);
      fclose(fpao);
      fclose(fma);
      fclose(fpe);
      fclose(fpea);
    }
  else if (data==1)//escribe el valor de las magnitudes en los respectivos ficheros.
    {
      fen = fopen(efile, "a");
      fpao = fopen(pfile, "a");
      fma = fopen(mfile, "a");
      fpe = fopen(pefile, "a");
      fpea = fopen(peafile, "a");
      fprintf(fen, "%f\t", set_energy());
      fprintf(fpao, "%f\t", orientacional());
      fprintf(fma, "%f\t", set_magnet());
      fprintf(fpe, "%f\t", set_perimeter(0));
      fprintf(fpea, "%f\t", set_perimeter(1));
      fclose(fen);
      fclose(fpao);
      fclose(fma);      
      fclose(fpe);
      fclose(fpea);
    }
  else if (data==2)//escribe un salto de linea
    { 
      fen = fopen(efile, "a");
      fpao = fopen(pfile, "a");
      fma = fopen(mfile, "a");
      fpe = fopen(pefile, "a");
      fpea = fopen(peafile, "a");
      fprintf(fen, "\n");
      fprintf(fpao, "\n");
      fprintf(fma, "\n");
      fprintf(fpe, "\n");
      fprintf(fpea, "\n");
      fclose(fen);
      fclose(fpao);
      fclose(fma);
      fclose(fpe);
      fclose(fpea);
    }
}


double set_perimeter(int pp)
{
  int i;
  double per=0,opi;
  
  
  for (i=0;i<N;i++)
    {
      opi=opposite_n(i,pp);
      if (opi==2) per+=1.4142;
      else if (opi==1) per++;
    }
    
  return (per/l/2);   
  
}


int opposite_n(int n1, int pp)
{
  int i, cnt=0;
  double fnn,av;

  if (pp==0)
    {
      fnn=fir[n1];

      if (fir[right[n1]]*fnn<0) cnt++;
      if (fir[left[n1]]*fnn<0) cnt++;
      if (fir[up[n1]]*fnn<0) cnt++;
      if (fir[down[n1]]*fnn<0) cnt++;
    }
  else
    {
      av=0;
      for (i=0;i<N;i++) av+=fir[i];
      av=av/N;

      fnn=fir[n1]-av;

      if ((fir[right[n1]]-av)*fnn<-.0001) cnt++;
      if ((fir[left[n1]]-av)*fnn<-.0001) cnt++;
      if ((fir[up[n1]]-av)*fnn<-.0001) cnt++;
      if ((fir[down[n1]]-av)*fnn<-.0001) cnt++;
      
    }

  return (cnt);

}
