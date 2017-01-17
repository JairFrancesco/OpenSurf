#include <opencv/cv.h>
#include <algorithm>
#include "utils.h"

class ImagenIntegral
{
	public:
		/* Calcula la imagen integral de la imagen img.  
		 * Asume que la imagen de origen es un punto flotante de 32 bits.
		 * Retorna un IplImage (OpenCV) en forma de 32 bits flotante
		*/
		IplImage* Calcular(IplImage *origen){

			IplImage *image = getGray(origen); 
			IplImage *imagenIntegral = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1); //Convertir la imagen a un solo canal de 32F
			int altura = image->height;
			int ancho = image->width;
			int step = image->widthStep/sizeof(float);
			float* datos   = (float *) image->imageData;  
			float* datosIntegrales = (float *) imagenIntegral->imageData;  

			float tmp = 0.0f;
			for(int j=0; j<ancho; j++) 
			{
				tmp += datos[j]; 
				datosIntegrales[j] = tmp;
			}		
			for(int i=1; i<altura; ++i) 
			{
				tmp = 0.0f;
				for(int j=0; j<ancho; ++j) 
				{
				  tmp += datos[i*step+j]; 
				  datosIntegrales[i*step+j] = tmp + datosIntegrales[(i-1)*step+j];
				}
			}
			cvReleaseImage(&image); //Liberar de memoria la imagen en escala de grises
			return imagenIntegral;
		}

		//Calcular la suma de pixeles dentro del rectangulo especificado por la coordenada de inicio superior izquierda y tamaño
		float CajaIntegral(IplImage *imagen, int fila, int columna, int filas, int columnas) 
		{
		  float*datos = (float *) imagen->imageData;
		  int step = imagen->widthStep/sizeof(float);
		  int r1 = std::min(fila,imagen->height) - 1;
		  int c1 = std::min(columna,imagen->width)  - 1;
		  int r2 = std::min(fila + filas,imagen->height) - 1;
		  int c2 = std::min(columna + columnas,imagen->width)  - 1;
		  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
		  if (r1 >= 0 && c1 >= 0) A = datos[r1 * step + c1];
		  if (r1 >= 0 && c2 >= 0) B = datos[r1 * step + c2];
		  if (r2 >= 0 && c1 >= 0) C = datos[r2 * step + c1];
		  if (r2 >= 0 && c2 >= 0) D = datos[r2 * step + c2];
		  return std::max(0.f, A - B - C + D);
		}
};

