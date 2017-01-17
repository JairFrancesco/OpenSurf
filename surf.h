
#ifndef SURF_H
#define SURF_H

#include <opencv/cv.h>
#include "Keypoint.h"
//#include "integral.h"
#include "ImagenIntegral.h"

#include <vector>

class Surf {
  
  public:
    
    // img es una imagen integral
    Surf(IplImage *img, std::vector<Keypoint> &ipts);

    //!Describir todas las caracteristicas en el vector dado
    void getDescriptors(bool bUpright = false);
  
  public:
    void getOrientation();
    void getDescriptor(bool bUpright = false);

    float gaussian(int x, int y, float sig);
    float gaussian(float x, float y, float sig);

    float haarX(int row, int column, int size);
    float haarY(int row, int column, int size);

    float getAng(float X, float Y);


    //! imagen integral donde los Keypoints han sido detectados
    IplImage *img;
    //! Keypoints vector
    KVector &kps;
    //! indice de actual Keypoint en el vector
    int index;
};


#endif