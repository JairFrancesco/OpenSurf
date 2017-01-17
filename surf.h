
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
  
  private:
    
    //---------------- Private Functions -----------------//

    //! Assign the current Keypoint an orientation
    void getOrientation();
    
    //! Get the descriptor. See Agrawal ECCV 08
    void getDescriptor(bool bUpright = false);

    //! Calculate the value of the 2d gaussian at x,y
    inline float gaussian(int x, int y, float sig);
    inline float gaussian(float x, float y, float sig);

    //! Calculate Haar wavelet responses in x and y directions
    inline float haarX(int row, int column, int size);
    inline float haarY(int row, int column, int size);

    //! Get the angle from the +ve x-axis of the vector given by [X Y]
    float getAngle(float X, float Y);


    //! imagen integral donde los Keypoints han sido detectados
    IplImage *img;
    //! Keypoints vector
    IpVec &ipts;
    //! indice de actual Keypoint en el vector
    int index;
};


#endif