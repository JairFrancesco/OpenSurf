
#include "Stitching.h"

/*

void drawKpoints(IplImage *img, vector<Keypoint> &ipts, int tailSize)
{
  Keypoint *ipt;
  float s, o;
  int r1, c1, r2, c2, lap;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    s = (2.5f * ipt->scale);
    o = ipt->orientacion;
    lap = ipt->laplacian;
    r1 = fRound(ipt->y);
    c1 = fRound(ipt->x);
    c2 = fRound(s * cos(o)) + c1;
    r2 = fRound(s * sin(o)) + r1;

    if (o) // Green line indicates orientation
      cvLine(img, cvPoint(c1, r1), cvPoint(c2, r2), cvScalar(0, 255, 0));
    else  // Green dot if using upright version
      cvCircle(img, cvPoint(c1,r1), 1, cvScalar(0, 255, 0),-1);

    if (lap == 1)
    { // Blue circles indicate dark blobs on light backgrounds
      cvCircle(img, cvPoint(c1,r1), flred(s), cvScalar(255, 0, 0),1);
    }
    else if (lap == 0)
    { // Red circles indicate light blobs on dark backgrounds
      cvCircle(img, cvPoint(c1,r1), flred(s), cvScalar(0, 0, 255),1);
    }
    else if (lap == 9)
    { // Red circles indicate light blobs on dark backgrounds
      cvCircle(img, cvPoint(c1,r1), flred(s), cvScalar(0, 255, 0),1);
    }

    // Draw motion from ipoint dx and dy
    if (tailSize)
    {
      cvLine(img, cvPoint(c1,r1),
        cvPoint(int(c1+ipt->dx*tailSize), int(r1+ipt->dy*tailSize)),
        cvScalar(255,255,255), 1);
    }
  }
}

int maino()
{
  //vector de kpoints
  std::cout<<"asdsadsa"<<std::endl;
  KVector kps;
    IplImage *img=cvLoadImage("Firefox_wallpaper.png");
  int index;
  ImagenIntegral* intimg;
  IplImage *int_img = intimg->Calcular(img);
  bool rot=false;
  int octaves=4;
  int inter=4;
  int init=2;
  float th=0.0001f;
  FastHessian fh(int_img, kps, octaves, inter, init, th);
 
    fh.getKeypoints();


    if (!kps.size()) return -1;

  int kpssize=(int)kps.size();

  if (rot)
  {
    for (int i=0; i < kpssize; ++i)
    {
        index=i;
        Descriptor(true, kps,img,index);
    }
  }
  else
  {
    for (int i = 0; i < kpssize; ++i)
    {
      index = i;
      Orientation(kps,index,img);
      Descriptor(false,kps,img,index);
    }
  }

    cvReleaseImage(&int_img);
  drawKpoints(img,kps,0);
  showImage(img);
}*/

int main()
{
  std::cout<<"llego"<<std::endl;
  Stitching* stitch = new Stitching();
  stitch->obtenerPanorama("panorama_image1.jpg", "panorama_image2.jpg");
  //cv::Mat image1= imread("panorama_image1.jpg");

  //Stitching nuevo;
  //nuevo.obtenerPanorama("panorama_image1.jpg","panorama_image2.jpg");
}
