#ifndef UTILS_H
#define UTILS_H

#include <opencv/cv.h>
#include "Keypoint.h"
#include <vector>


//! Display error message and terminate program
void error(const char *msg);

//! Show the provided image and wait for keypress
void showImage(const IplImage *img);

//! Show the provided image in titled window and wait for keypress
void showImage(char *title,const IplImage *img);

// Convert image to single channel 32F
IplImage* getGray(const IplImage *img);

//! Draw a single feature on the image
void drawKeypoint(IplImage *img, Keypoint &ipt, int tailSize = 0);

//! Draw all the Ipoints in the provided vector
void drawKeypoints(IplImage *img, std::vector<Keypoint> &ipts, int tailSize = 0);

//! Draw descriptor windows around Ipoints in the provided vector
void drawWindows(IplImage *img, std::vector<Keypoint> &ipts);

// Draw the FPS figure on the image (requires at least 2 calls)
void drawFPS(IplImage *img);

//! Draw a Point at feature location
void drawPoint(IplImage *img, Keypoint &ipt);

//! Draw a Point at all features
void drawPoints(IplImage *img, std::vector<Keypoint> &ipts);

//! Save the SURF features to file
void saveSurf(char *filename, std::vector<Keypoint> &ipts);

//! Load the SURF features from file
void loadSurf(char *filename, std::vector<Keypoint> &ipts);

//! Round float to nearest integer
inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}

#endif
