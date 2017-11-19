#include "stdafx.h"
#include "opencv2/core/core.hpp"      // Main OpenCV functions
#include "opencv2/highgui/highgui.hpp" // GUI functions
#include <math.h>
#include <vector>
#include <iostream>


using namespace cv;
using namespace std;

void Gaussian(Mat image, float mean, float stddev); // Gaussian noise 
void Gaussian2(Mat image, float mean, float stddev); // Gaussian noise 2
double CentralLimitTheorem(float mean, float variance); // Even distribution generator
void RankedOrderFilter(Mat image, int x, int y, int rank); // Ranked order filter
void TrimmedMean(Mat image, int x, int y, float a); // Trimmed mean filter
void SobelEdgeDetector(Mat image, int x); // Sobel edge detector
void PrewittEdgeDetector(Mat image, int x); // Prewitt edge detector
Mat genSobel(int x); // Sobel mask
Mat genPrewitt(int x); // Prewitt mask
Mat conv2d(Mat signal, Mat filter); // 2D convolution
void AdaptiveThreshold(Mat image, int x, int y, int threshold); // Adaptive threshold segmentation

void Display(Mat image, char* s); // Display image

void InsertNoise(Mat image);
void RemoveNoise(Mat image);
void EdgeDetection(Mat image);
void ImageSegmentation(Mat image);

void read();
string s;
Mat image;

int main(int argc, char** argv)
{
	read();
	char ans;

	while (1)
	{
		cout << "\nWhat shall I do with the image? Type:\n(1) to insert noise\n(2) to remove noise\n(3) to detect edges\n(4) for image segmentation\n(5)Read new image\n(0) to exit\n";
		cin >> ans;

		if (ans == '1')
		{
			InsertNoise(image);
		}
		else if (ans == '2')
		{
			Display(image, "original image");
			Mat img = image.clone();
			RemoveNoise(img);
		}
		else if (ans == '3')
		{
			EdgeDetection(image);
		}
		else if (ans == '4')
		{
			ImageSegmentation(image);
		}
		else if (ans == '5')
		{
			read();
		}
		else if (ans == '0')
		{
			break;
		}

		waitKey(0);
	}

	system("pause");
	return 0;
}

void read()
{
	do
	{
		cout << "Type image source (*.jpg):";
		cin >> s;
		image = imread(s, CV_8U);
	} while (!image.data);
}

void Gaussian(Mat image, float mean, float stddev)
{
	Mat noise(image.rows, image.cols, image.type());
	randn(noise, mean, stddev);
	image += noise;
}

void Gaussian2(Mat image, float mean, float stddev)
{
	//Mat noise(image.rows,image.cols,image.type());
	for (int i = 0; i<image.rows; i++)
	{
		for (int j = 0; j<image.cols; j++)
		{
			//image.at<cv::Vec3b>(i,j);
			image.at<cv::Vec3b>(i, j)[0] += (uchar)CentralLimitTheorem(mean, stddev);
			image.at<cv::Vec3b>(i, j)[1] += (uchar)CentralLimitTheorem(mean, stddev);
			image.at<cv::Vec3b>(i, j)[2] += (uchar)CentralLimitTheorem(mean, stddev);
		}
	}
	//image+=noise;	
}

double CentralLimitTheorem(float mean, float stddev)
{
#define NSUM 25
	//#define RAND_MAX 1

	double x = 0;
	int i;
	for (i = 0; i < NSUM; i++)
		x += (double)rand() / (RAND_MAX + 1.0);
	x -= NSUM / 2.0;   /* set mean to 0 */
	x /= sqrt(NSUM / 12.0); /* adjust variance to 1 */

	x = mean + stddev*x;

	return x;
}

void InsertNoise(Mat image)
{
	float mean, stddev;
	cout << "Mean: ";
	cin >> mean;
	cout << "Standard Deviation: ";
	cin >> stddev;

	//Mat output = image.clone();
	Gaussian(image, mean, stddev);
	//Gaussian2(image,mean,stddev);

	Display(image, "noise");
}

void RemoveNoise(Mat image)
{
	char filter;
	int h, w, r;
	float a;
	cout << "Which filter should be applied?\n(a)Ranked Order Filter\n(b)Trimmed Mean\n(a/b)? ";
	cin >> filter;
	cout << endl;
	if (filter == 'a')
	{
		cout << "Mask size?\nheight: ";
		cin >> h;
		cout << "width: ";
		cin >> w;
		cout << "rank: ";
		cin >> r;
		cout << endl;
		RankedOrderFilter(image, h, w, r);
	}
	if (filter == 'b')
	{
		cout << "Mask size?\nheight: ";
		cin >> h;
		cout << "width: ";
		cin >> w;
		cout << "a: ";
		cin >> a;
		cout << endl;
		TrimmedMean(image, h, w, a);
	}

	Display(image, "filtered noise");
}

void RankedOrderFilter(Mat image, int x, int y, int rank) //x,y perittoi
{
	image.convertTo(image, CV_32F);
	//cout<<image.type();
	for (int channel = 0; channel<image.channels(); channel++) //gia kathe kanali
	{
		for (int i = 0; i<image.rows; i++) //gia kathe grammi
		{
			for (int j = 0; j<image.cols; j++) //gia kathe stili
			{
				vector<float> xi;
				for (int k = i - x / 2; k <= i + x / 2; k++) //efarmogi parathyrou x*y
				{
					for (int m = j - y / 2; m <= j + y / 2; m++)
					{
						if ((k<0) || (m<0)) xi.push_back(0);
						else xi.push_back(image.data[image.step[0] * i + image.step[1] * j + channel]);
					}
				}
				std::sort(std::begin(xi), std::end(xi));
				image.data[image.step[0] * i + image.step[1] * j + channel] = xi[rank];
			}
		}
	}
	image.convertTo(image, CV_8U);
}

void TrimmedMean(Mat image, int x, int y, float a)
{
	image.convertTo(image, CV_32F);
	for (int channel = 0; channel<image.channels(); channel++) //gia kathe kanali
	{
		for (int i = 0; i<image.rows; i++) //gia kathe grammi
		{
			for (int j = 0; j<image.cols; j++) //gia kathe stili
			{
				std::vector<double> xi;
				for (int k = i - x / 2; k <= i + x / 2; k++)
				{
					for (int m = j - y / 2; m <= j + y / 2; m++)
					{
						if ((k<0) || (m<0)) xi.push_back(0);
						else xi.push_back(image.data[image.step[0] * i + image.step[1] * j + channel]);
					}
				}

				std::sort(std::begin(xi), std::end(xi));
				int n = xi.size();
				double sum = 0;
				for (int k = ((a*n) * 10) / 10; k<((n - a*n) * 10) / 10; k++)
				{
					sum += xi[k];
				}
				if (a != 0.5) sum /= (n*(1 - 2 * a));

				image.data[image.step[0] * i + image.step[1] * j + channel] = (int)sum;

			}
		}
	}
	image.convertTo(image, CV_8U);
}

void EdgeDetection(Mat image)
{
	char filter;
	int x;
	cout << "Sobel/Prewitt? (a/b)? :";
	cin >> filter;
	if (filter == 'a')
	{
		cout << "Sobel Dimensions?";
		cout << "heigth x: "; cin >> x;
		SobelEdgeDetector(image, x);
	}
	else
	{
		cout << "Prewitt Dimensions?";
		cout << "heigth x: "; cin >> x;

		PrewittEdgeDetector(image, x);
	}

}

void SobelEdgeDetector(Mat image, int x)
{

	Mat sobelX = genSobel(x);
	Mat sobelY; transpose(sobelX, sobelY);
	Mat edgesX = conv2d(image, sobelX);
	Mat edgesY = conv2d(image, sobelY);
	Mat edges(image.rows, image.cols, DataType<float>::type);

	for (int i = 0; i<image.rows; i++)
		for (int j = 0; j<image.cols; j++)
		{
			edges.at<float>(i, j) = (float)sqrt(pow(edgesX.at<float>(i, j), 2) + pow(edgesY.at<float>(i, j), 2));
		}
	Display(edges, "sobel edges");
	Display(edgesX, "sobel X edges");
	Display(edgesY, "sobel Y edges");

}

Mat genSobel(int x)
{
	int a3[3][3] = { { -1,0,1 },
	{ -2,0,2 },
	{ -1,0,1 } };
	int a5[5][5] = { { -2,-1,0,1,2 },
	{ -3,-2,0,2,3 },
	{ -4,-3,0,3,4 },
	{ -3,-2,0,2,3 },
	{ -2,-1,0,1,2 } };
	int a7[7][7] = { { -3,-2,-1,0,1,2,3 },
	{ -4,-3,-2,0,2,3,4 },
	{ -5,-4,-3,0,3,4,5 },
	{ -6,-5,-4,0,4,5,6 },
	{ -5,-4,-3,0,3,4,5 },
	{ -4,-3,-2,0,2,3,4 },
	{ -3,-2,-1,0,1,2,3 } };
	int a9[9][9] = { { -4,-3,-2,-1,0,1,2,3,4 },
	{ -5,-4,-3,-2,0,2,3,4,5 },
	{ -6,-5,-4,-3,0,3,4,5,6 },
	{ -7,-6,-5,-4,0,4,5,6,7 },
	{ -8,-7,-6,-5,0,5,6,7,8 },
	{ -7,-6,-5,-4,0,4,5,6,7 },
	{ -6,-5,-4,-3,0,3,4,5,6 },
	{ -5,-4,-3,-2,0,2,3,4,5 },
	{ -4,-3,-2,-1,0,1,2,3,4 } };
	Mat a;
	if (x == 3)
	{
		a = Mat(3, 3, DataType<int>::type);
		for (int i = 0; i<x; i++)
			for (int j = 0; j<x; j++) {
				a.at<int>(i, j) = a3[i][j];
			}
	}
	else if (x == 5)
	{
		a = Mat(5, 5, DataType<int>::type);
		for (int i = 0; i<x; i++)
			for (int j = 0; j<x; j++) {
				a.at<int>(i, j) = a5[i][j];
			}
	}
	else if (x == 7)
	{
		a = Mat(7, 7, DataType<int>::type);
		for (int i = 0; i<x; i++)
			for (int j = 0; j<x; j++) {
				a.at<int>(i, j) = a7[i][j];
			}
	}
	else
	{
		a = Mat(9, 9, DataType<int>::type);
		for (int i = 0; i<x; i++)
			for (int j = 0; j<x; j++) {
				a.at<int>(i, j) = a9[i][j];
			}
	}

	return a;
}

Mat genPrewitt(int x)
{
	int a3[3][3] = { { -1,0,1 },
	{ -1,0,1 },
	{ -1,0,1 } };
	Mat a = Mat(3, 3, DataType<int>::type);
	for (int i = 0; i<x; i++)
		for (int j = 0; j<x; j++) {
			a.at<int>(i, j) = a3[i][j];
		}
	return a;
}

void PrewittEdgeDetector(Mat image, int x)
{
	Mat prewittX = genPrewitt(x);
	Mat prewittY; transpose(prewittX, prewittY);
	Mat edgesX = conv2d(image, prewittX);
	Mat edgesY = conv2d(image, prewittY);
	Mat edges(image.rows, image.cols, DataType<float>::type);

	for (int i = 0; i<image.rows; i++)
		for (int j = 0; j<image.cols; j++)
		{
			edges.at<float>(i, j) = (float)sqrt(pow(edgesX.at<float>(i, j), 2) + pow(edgesY.at<float>(i, j), 2));
		}
	Display(edges, "sobel edges");
	Display(edgesX, "sobel X edges");
	Display(edgesY, "sobel Y edges");
}

Mat conv2d(Mat signal, Mat filter)
{
	int rows = signal.rows;
	int cols = signal.cols;
	int m1 = filter.rows;
	int m2 = filter.cols;

	Mat conv(rows, cols, DataType<float>::type);
	for (int n1 = 0; n1<rows; n1++)
	{
		for (int n2 = 0; n2<cols; n2++)
		{
			float sum = 0;
			for (int k1 = 0; k1<m1; k1++)
			{
				for (int k2 = 0; k2<m2; k2++)
				{
					if ((n1 - k1 >= 0) && (n2 - k2 >= 0))
					{
						sum += (filter.at<int>(k1, k2))*(signal.data[signal.step[0] * (n1 - k1) + signal.step[1] * (n2 - k2)]);
					}
				}
			}
			conv.at<float>(n1, n2) = sum;
		}
	}
	//conv.convertTo(conv,CV_8U);
	return conv;
}

void AdaptiveThreshold(Mat image, int x, int y, int threshold)
{
	int rows = image.rows;
	int cols = image.cols;

	Mat segment(rows, cols, DataType<uchar>::type);
	for (int n1 = 0; n1<rows; n1++)
	{
		for (int n2 = 0; n2<cols; n2++)
		{
			float sum = 0;
			for (int k1 = 0; k1<x; k1++)
			{
				for (int k2 = 0; k2<y; k2++)
				{
					if ((n1 - k1 >= 0) && (n2 - k2 >= 0))
					{
						sum += (image.data[image.step[0] * (n1 - k1) + image.step[1] * (n2 - k2)]);
					}
				}
			}
			sum /= (float)(x*y);
			if (sum<threshold) sum = 0;
			else { sum = 255; }
			segment.at<uchar>(n1, n2) = sum;
		}
	}
	Display(segment, "segmented image");
}

void ImageSegmentation(Mat image)
{
	int x, y, t;
	cout << "Adaptive Threshold\nNeighborhood Region?\nx(height)?: ";
	cin >> x;
	cout << "y(width)?: ";
	cin >> y;
	cout << "Threshold?: ";
	cin >> t;
	AdaptiveThreshold(image, x, y, t);
}

void Display(Mat image, char* s)
{
	if (image.type() != CV_8U) image.convertTo(image, CV_8U);
	cvNamedWindow(s, CV_WINDOW_AUTOSIZE);
	imshow(s, image);
	waitKey(30);
	//image.convertTo(image,CV_32F);
}

/*IplImage* img = cvLoadImage( "merkel.jpg" ); //change the name (image.jpg) according to your Image filename.
cvNamedWindow( "helloWorld", CV_WINDOW_AUTOSIZE );
cvShowImage("helloWorld", img);
cvWaitKey(0);
cvReleaseImage( &img );
cvDestroyWindow( "helloWorld" );*/