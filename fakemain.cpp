#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/calib3d.hpp>
#include "integral.h"

#include "fasthessian.h"

#include <vector>
#include <math.h>

//-------------------------------------------------------
/*
class Ipoint; // Pre-declaration
typedef std::vector<Ipoint> IpVec;
typedef std::vector<std::pair<Ipoint, Ipoint> > IpPairVec;

//-------------------------------------------------------

//! Ipoint operations
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches);
int translateCorners(IpPairVec &matches, const CvPoint src_corners[4], CvPoint dst_corners[4]);

void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches)
{
  float dist, d1, d2;
  Ipoint *match;

  matches.clear();

  for(unsigned int i = 0; i < ipts1.size(); i++) 
  {
    d1 = d2 = FLT_MAX;

    for(unsigned int j = 0; j < ipts2.size(); j++) 
    {
      dist = ipts1[i] - ipts2[j];  

      if(dist<d1) // if this feature matches better than current best
      {
        d2 = d1;
        d1 = dist;
        match = &ipts2[j];
      }
      else if(dist<d2) // this feature matches better than second best
      {
        d2 = dist;
      }
    }

    // If match has a d1:d2 ratio < 0.65 ipoints are a match
    if(d1/d2 < 0.65) 
    { 
      // Store the change in position
      ipts1[i].dx = match->x - ipts1[i].x; 
      ipts1[i].dy = match->y - ipts1[i].y;
      matches.push_back(std::make_pair(ipts1[i], *match));
    }
  }
}

int translateCorners(IpPairVec &matches, const CvPoint src_corners[4], CvPoint dst_corners[4])
{
  double h[9];
  cv::Mat _h = cv::Mat(3, 3, CV_64F, h);
  std::vector<CvPoint2D32f> pt1, pt2;
  cv::Mat _pt1, _pt2;
  
  int n = (int)matches.size();
  if( n < 4 ) return 0;

  // Set vectors to correct size
  pt1.resize(n);
  pt2.resize(n);

  // Copy Ipoints from match vector into cvPoint vectors
  for(int i = 0; i < n; i++ )
  {
    pt1[i] = cvPoint2D32f(matches[i].second.x, matches[i].second.y);
    pt2[i] = cvPoint2D32f(matches[i].first.x, matches[i].first.y);
  }
  _pt1 = cv::Mat(1, n, CV_32FC2, &pt1[0] );
  _pt2 = cv::Mat(1, n, CV_32FC2, &pt2[0] );

  cv::InputArray _pt1_ia(_pt1);
  cv::InputArray _pt2_ia(_pt2);
  cv::OutputArray _h_ia(_h);

  cv::Mat hMat = cv::findHomography(_pt1_ia, _pt2_ia, _h_ia, CV_RANSAC, 5.0);
   // Find the homography (transformation) between the two sets of points
  if(hMat.empty())  // this line requires opencv 1.1
    return 0;

  // Translate src_corners to dst_corners using homography
  for(int i = 0; i < 4; i++ )
  {
    double x = src_corners[i].x, y = src_corners[i].y;
    double Z = 1./(h[6]*x + h[7]*y + h[8]);
    double X = (h[0]*x + h[1]*y + h[2])*Z;
    double Y = (h[3]*x + h[4]*y + h[5])*Z;
    dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
  }
}


//-------------------------------------------------------

class Ipoint {

public:

  //! Destructor
  ~Ipoint() {};

  //! Constructor
  Ipoint() : orientation(0) {};

  //! Gets the distance in descriptor space between Ipoints
  float operator-(const Ipoint &rhs)
  {
    float sum=0.f;
    for(int i=0; i < 64; ++i)
      sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
    return sqrt(sum);
  };

  //! Coordinates of the detected interest point
  float x, y;

  //! Detected scale
  float scale;

  //! Orientation measured anti-clockwise from +ve x-axis
  float orientation;

  //! Sign of laplacian for fast matching purposes
  int laplacian;

  //! Vector of descriptor components
  float descriptor[64];

  //! Placeholds for point motion (can be used for frame to frame motion analysis)
  float dx, dy;

  //! Used to store cluster index
  int clusterIndex;
};
*/

const float pi = 3.14159f;

//! lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const double gauss25 [7][7] = {
  0.02546481, 0.02350698, 0.01849125, 0.01239505, 0.00708017, 0.00344629, 0.00142946,
  0.02350698, 0.02169968, 0.01706957, 0.01144208, 0.00653582, 0.00318132, 0.00131956,
  0.01849125, 0.01706957, 0.01342740, 0.00900066, 0.00514126, 0.00250252, 0.00103800,
  0.01239505, 0.01144208, 0.00900066, 0.00603332, 0.00344629, 0.00167749, 0.00069579,
  0.00708017, 0.00653582, 0.00514126, 0.00344629, 0.00196855, 0.00095820, 0.00039744,
  0.00344629, 0.00318132, 0.00250252, 0.00167749, 0.00095820, 0.00046640, 0.00019346,
  0.00142946, 0.00131956, 0.00103800, 0.00069579, 0.00039744, 0.00019346, 0.00008024
};

float getAngle(float X, float Y)
{
  if(X > 0 && Y >= 0)
    return atan(Y/X);

  if(X < 0 && Y >= 0)
    return pi - atan(-Y/X);

  if(X < 0 && Y < 0)
    return pi + atan(Y/X);

  if(X > 0 && Y < 0)
    return 2*pi - atan(-Y/X);

  return 0;
}


inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}



float gaussian(float x, float y, float sig)
{
  return 1.0f/(2.0f*pi*sig*sig) * exp( -(x*x+y*y)/(2.0f*sig*sig));
}

float gaussian(int x, int y, float sig)
{
  return (1.0f/(2.0f*pi*sig*sig)) * exp( -(x*x+y*y)/(2.0f*sig*sig));
}




float haarX(int row, int column, int s, IplImage *img)
{
  return BoxIntegral(img, row-s/2, column, s, s/2)-1 * BoxIntegral(img, row-s/2, column-s/2, s, s/2);
}

float haarY(int row, int column, int s, IplImage *img)
{
  return BoxIntegral(img, row, column-s/2, s/2, s)-1 * BoxIntegral(img, row-s/2, column-s/2, s/2, s);
}


float Orientation(IpVec &ipts,int & index,IplImage* & img)
{
  Ipoint *ipt = &(ipts[index]);
  float gauss = 0.f, scale = ipt->scale;
  const int s = fRound(scale), r = fRound(ipt->y), c = fRound(ipt->x);
  std::vector<float> resX(109), resY(109), Ang(109);
  const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

  int idx = 0;
  // calculate haar responses for points within radius of 6*scale
  for(int i = -6; i <= 6; ++i) 
  {
    for(int j = -6; j <= 6; ++j) 
    {
      if(i*i + j*j < 36) 
      {
        gauss = static_cast<float>(gauss25[id[i+6]][id[j+6]]);  // could use abs() rather than id lookup, but this way is faster
        resX[idx] = gauss * haarX(r+j*s, c+i*s, 4*s,img);
        resY[idx] = gauss * haarY(r+j*s, c+i*s, 4*s,img);
        Ang[idx] = getAngle(resX[idx], resY[idx]);
        ++idx;
      }
    }
  }

  // calculate the dominant direction 
  float sumX=0.f, sumY=0.f;
  float max=0.f, orientation = 0.f;
  float ang1=0.f, ang2=0.f;

  // loop slides pi/3 window around feature point
  for(ang1 = 0; ang1 < 2*pi;  ang1+=0.15f) {
    ang2 = ( ang1+pi/3.0f > 2*pi ? ang1-5.0f*pi/3.0f : ang1+pi/3.0f);
    sumX = sumY = 0.f; 
    for(unsigned int k = 0; k < Ang.size(); ++k) 
    {
      // get angle from the x-axis of the sample point
      const float & ang = Ang[k];

      // determine whether the point is within the window
      if (ang1 < ang2 && ang1 < ang && ang < ang2) 
      {
        sumX+=resX[k];  
        sumY+=resY[k];
      } 
      else if (ang2 < ang1 && 
        ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2*pi) )) 
      {
        sumX+=resX[k];  
        sumY+=resY[k];
      }
    }

    // if the vector produced from this window is longer than all 
    // previous vectors then this forms the new dominant direction
    if (sumX*sumX + sumY*sumY > max) 
    {
      // store largest orientation
      max = sumX*sumX + sumY*sumY;
      orientation = getAngle(sumX, sumY);
    }
  }

  // assign orientation of the dominant response vector
  ipt->orientation = orientation;
}




void Descriptor(bool bUpright,IpVec &ipts, IplImage* & img, int & index)
{
  int y, x, sample_x, sample_y, count=0;
  int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
  float scale, *desc, dx, dy, mdx, mdy, co, si;
  float gauss_s1 = 0.f, gauss_s2 = 0.f;
  float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
  float cx = -0.5f, cy = 0.f; //Subregion centers for the 4x4 gaussian weighting

  Ipoint *ipt = &ipts[index];
  scale = ipt->scale;
  x = fRound(ipt->x);
  y = fRound(ipt->y);  
  desc = ipt->descriptor;

  if (bUpright)
  {
    co = 1;
    si = 0;
  }
  else
  {
    co = cos(ipt->orientation);
    si = sin(ipt->orientation);
  }

  i = -8;

  //Calculate descriptor for this interest point
  while(i < 12)
  {
    j = -8;
    i = i-4;

    cx += 1.f;
    cy = -0.5f;

    while(j < 12) 
    {
      dx=dy=mdx=mdy=0.f;
      cy += 1.f;

      j = j - 4;

      ix = i + 5;
      jx = j + 5;

      xs = fRound(x + ( -jx*scale*si + ix*scale*co));
      ys = fRound(y + ( jx*scale*co + ix*scale*si));

      for (int k = i; k < i + 9; ++k) 
      {
        for (int l = j; l < j + 9; ++l) 
        {
          //Get coords of sample point on the rotated axis
          sample_x = fRound(x + (-l*scale*si + k*scale*co));
          sample_y = fRound(y + ( l*scale*co + k*scale*si));

          //Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5f*scale);
          rx = haarX(sample_y, sample_x, 2*fRound(scale),img);
          ry = haarY(sample_y, sample_x, 2*fRound(scale),img);

          //Get the gaussian weighted x and y responses on rotated axis
          rrx = gauss_s1*(-rx*si + ry*co);
          rry = gauss_s1*(rx*co + ry*si);

          dx += rrx;
          dy += rry;
          mdx += fabs(rrx);
          mdy += fabs(rry);

        }
      }

      //Add the values to the descriptor vector
      gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

      desc[count++] = dx*gauss_s2;
      desc[count++] = dy*gauss_s2;
      desc[count++] = mdx*gauss_s2;
      desc[count++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;

      j += 9;
    }
    i += 9;
  }

  //Convert to Unit Vector
  len = sqrt(len);
  for(int i = 0; i < 64; ++i)
    desc[i] /= len;
}

int main()
{
  //vector de kpoints
  std::cout<<"asdsadsa"<<std::endl;
  IpVec ipts;
    IplImage *img=cvLoadImage("Firefox_wallpaper.png");
  int index;
  IplImage *int_img = Integral(img);
  bool upright = false; 
  int octaves = 4;
  int intervals = 4;
  int init_sample = 2;
  float thres = 0.0001f;
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);
 
  // Extract interest points and store in vector ipts
    fh.getIpoints();


    ///
    if (!ipts.size()) return -1;

  // Get the size of the vector for fixed loop bounds
  int ipts_size = (int)ipts.size();

  if (upright)
  {
    // U-SURF loop just gets descriptors
    for (int i = 0; i < ipts_size; ++i)
    {
      // Set the Ipoint to be described
        index = i;

        // Extract upright (i.e. not rotation invariant) descriptors
        Descriptor(true, ipts,img,index);
    }
  }
  else
  {
    // Main SURF-64 loop assigns orientations and gets descriptors
    for (int i = 0; i < ipts_size; ++i)
    {
      // Set the Ipoint to be described
      index = i;

      // Assign Orientations and extract rotation invariant descriptors
      Orientation(ipts,index,img);
      Descriptor(false,ipts,img,index);
    }
  }
    ///




  // Extract the descriptors for the ipts
//  des.getDescriptors(upright);

  // Deallocate the integral image
    cvReleaseImage(&int_img);
}
