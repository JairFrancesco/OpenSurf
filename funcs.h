
const float pi = 3.14159f;
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


		int flred(float flt)
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
		  ImagenIntegral* inte;
		  return inte->CajaIntegral(img, row-s/2, column, s, s/2)-1 * inte->CajaIntegral(img, row-s/2, column-s/2, s, s/2);
		}

		float haarY(int row, int column, int s, IplImage *img)
		{
		  ImagenIntegral* inte;
		  return inte->CajaIntegral(img, row, column-s/2, s/2, s)-1 * inte->CajaIntegral(img, row-s/2, column-s/2, s/2, s);
		}


		void Orientation(KVector &kps1,int & index,IplImage* & img)
		{
		  Keypoint *kp = &(kps1[index]);
		  float gauss = 0.f, scale = kp->scale;
		  const int s = flred(scale), r = flred(kp->y), c = flred(kp->x);
		  std::vector<float> resX(109), resY(109), Ang(109);
		  const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};
		  int idx = 0;
		  for(int i = -6; i <= 6; ++i) 
		  {
		    for(int j = -6; j <= 6; ++j) 
		    {
		      if(i*i + j*j < 36) 
		      {
		        gauss = static_cast<float>(gauss25[id[i+6]][id[j+6]]);  
		        resX[idx] = gauss * haarX(r+j*s, c+i*s, 4*s,img);
		        resY[idx] = gauss * haarY(r+j*s, c+i*s, 4*s,img);
		        Ang[idx] = getAngle(resX[idx], resY[idx]);
		        ++idx;
		      }
		    }
		  }
		  float sumX=0.f, sumY=0.f;
		  float max=0.f, orien = 0.f;
		  float ang1=0.f, ang2=0.f;

		  for(ang1 = 0; ang1 < 2*pi;  ang1+=0.15f) {
		    ang2 = ( ang1+pi/3.0f > 2*pi ? ang1-5.0f*pi/3.0f : ang1+pi/3.0f);
		    sumX = sumY = 0.f; 
		    for(unsigned int k = 0; k < Ang.size(); ++k) 
		    {
		      const float & ang = Ang[k];
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

		    if (sumX*sumX + sumY*sumY > max) 
		    {
		      // store largest orientation
		      max = sumX*sumX + sumY*sumY;
		      orien = getAngle(sumX, sumY);
		    }
		  }
		  kp->orientacion = orien;
		}




		void Descriptor(bool bUpright,KVector &kps, IplImage* & img, int & index)
		{
		  int y, x, sample_x, sample_y, count=0;
		  int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
		  float scale, *desc, dx, dy, mdx, mdy, co, si;
		  float gauss_s1 = 0.f, gauss_s2 = 0.f;
		  float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
		  float cx = -0.5f, cy = 0.f; 

		  Keypoint *kp = &kps[index];
		  scale = kp->scale;
		  x = flred(kp->x);
		  y = flred(kp->y);  
		  desc = kp->descriptor;

		  if (bUpright)
		  {
		    co = 1;
		    si = 0;
		  }
		  else
		  {
		    co = cos(kp->orientacion);
		    si = sin(kp->orientacion);
		  }

		  i = -8;

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

		      xs = flred(x +(-jx*scale*si + ix*scale*co));
		      ys = flred(y +(jx*scale*co + ix*scale*si));

		      for (int k = i; k < i + 9; ++k) 
		      {
		        for (int l = j; l < j + 9; ++l) 
		        {
		          sample_x = flred(x + (-l*scale*si + k*scale*co));
		          sample_y = flred(y + ( l*scale*co + k*scale*si));
		          gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5f*scale);
		          rx = haarX(sample_y, sample_x, 2*flred(scale),img);
		          ry = haarY(sample_y, sample_x, 2*flred(scale),img);
		          rrx = gauss_s1*(-rx*si + ry*co);
		          rry = gauss_s1*(rx*co + ry*si);

		          dx += rrx;
		          dy += rry;
		          mdx += fabs(rrx);
		          mdy += fabs(rry);

		        }
		      }

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

		  len = sqrt(len);
		  for(int i = 0; i < 64; ++i)
		    desc[i] /= len;
		}

