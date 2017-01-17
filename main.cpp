/*********************************************************** 
*  --- OpenSURF ---                                       *
*  This library is distributed under the GNU GPL. Please   *
*  use the contact form at http://www.chrisevansdev.com    *
*  for more information.                                   *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/

#include "surflib.h"
#include "kmeans.h"
#include <ctime>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#define PROCEDURE 5

using namespace cv;

//-------------------------------------------------------

//Dar la ruta de una imagen estatica
int mainImage()
{
  // Declare Ipoints and other stuff
  IpVec ipts;
  IplImage *img=cvLoadImage("Arequipa_01.jpg");

  // Detect and describe interest points in the image
  clock_t start = clock();
  surfDetDes(img, ipts, false, 5, 4, 2, 0.0004f); 
  clock_t end = clock();

  std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
  std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;

  // Draw the detected points
  drawIpoints(img, ipts);
  
  // Display the result
  showImage(img);

  return 0;
}

//Para mostrar coincidencias entre dos imagenes estaticas
int mainStaticMatch()
{
  IplImage *img1, *img2;
  img1 = cvLoadImage("panorama_image1.jpg");
  img2 = cvLoadImage("panorama_image2.jpg");

  cv::Mat image1= imread( "panorama_image2.jpg" );
  cv::Mat image2= imread( "panorama_image1.jpg" );

  IpVec ipts1, ipts2;
  surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
  surfDetDes(img2,ipts2,false,4,4,2,0.0001f);

  IpPairVec matches;
  getMatches(ipts1,ipts2,matches);

  //IpPairVec good_matches;

  //kmeans* kmean = new kmeans();
  //kmean->Distance(matches[i].first, matches[i].second);

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
  //Utilizar la matriz de homografía para deformar las imágenes
  cv::Mat result; //Almacena la imagen panoramica de resultado
  warpPerspective(image1,result,hMat,cv::Size(image1.cols+image2.cols,image1.rows));
  cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
  image2.copyTo(half); //
  imshow( "Result", result );


  /*
  for( int i = 0; i < descriptors_object.rows; i++ )
  { 
    if( kmean->Distance(matches[i].first, matches[i].second) < 3*min_dist )
    { 
      good_matches.push_back( matches[i]); 
    }
  }
  */

  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }

  std::cout<< "Matches: " << matches.size();

  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2",img2);
  cvWaitKey(0);

  return 0;
}

//-------------------------------------------------------

//-------------------------------------------------------

int main() 
{
  if (PROCEDURE == 1) return mainImage();
  if (PROCEDURE == 5) return mainStaticMatch();
}
