#ifndef SURFLIB_H
#define SURFLIB_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

//#include "integral.h"
#include "ImagenIntegral.h"
#include "fasthessian.h"
#include "surf.h"
#include "Keypoint.h"
#include "utils.h"


//! Library function builds vector of described interest points
inline void surfDetDes(IplImage *img,  /* image to find Keypoint in */
                       std::vector<Keypoint> &ipts, /* reference to vector of Keypoint */
                       bool upright = false, /* run in rotation invariant mode? */
                       int octaves = OCTAVES, /* number of octaves to calculate */
                       int intervals = INTERVALS, /* number of intervals per octave */
                       int init_sample = INIT_SAMPLE, /* initial sampling step */
                       float thres = THRES /* blob response threshold */)
{
  ImagenIntegral* imgInt = new ImagenIntegral();
  // Create integral-image representation of the image
  IplImage *int_img = imgInt->Calcular(img);
  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);
  // Extract interest points and store in vector ipts
  fh.getKeypoints();
  // Create Surf Descriptor Object
  Surf des(int_img, ipts);
  // Extract the descriptors for the ipts
  des.getDescriptors(upright);
  // Deallocate the integral image
  cvReleaseImage(&int_img);
}


//! Library function builds vector of interest points
inline void surfDet(IplImage *img,  /* image to find Keypoint in */
                    std::vector<Keypoint> &ipts, /* reference to vector of Keypoint */
                    int octaves = OCTAVES, /* number of octaves to calculate */
                    int intervals = INTERVALS, /* number of intervals per octave */
                    int init_sample = INIT_SAMPLE, /* initial sampling step */
                    float thres = THRES /* blob response threshold */)
{
  ImagenIntegral* imgInt = new ImagenIntegral();
  // Create integral image representation of the image
  IplImage *int_img = imgInt->Calcular(img);

  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

  // Extract interest points and store in vector ipts
  fh.getKeypoints();

  // Deallocate the integral image
  cvReleaseImage(&int_img);
}




//! Library function describes interest points in vector
inline void surfDes(IplImage *img,  /* image to find Keypoint in */
                    std::vector<Keypoint> &ipts, /* reference to vector of Keypoint */
                    bool upright = false) /* run in rotation invariant mode? */
{ 
  ImagenIntegral* imgInt = new ImagenIntegral();
  // Create integral image representation of the image
  IplImage *int_img = imgInt->Calcular(img);

  // Create Surf Descriptor Object
  Surf des(int_img, ipts);

  // Extract the descriptors for the ipts
  des.getDescriptors(upright);
  
  // Deallocate the integral image
  cvReleaseImage(&int_img);
}


#endif