// ImageAnalysis.cpp: implementation of the CImageAnalysis class.
//
//////////////////////////////////////////////////////////////////////


#include "ImageAnalysis.h"
#include "Util.h"
#include "Pixel.h"
#include <opencv2/imgproc.hpp>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

double CImageAnalysis::getSpatialMoment(cv::Moments moment, int x_order, int y_order)
{
	if (x_order + y_order > 3) {
		return NULL;
	}

	if (x_order == 0) {
		if (y_order == 0)
			return moment.m00;
		else if (y_order == 1)
			return moment.m01;
		else if (y_order == 2)
			return moment.m02;
		else
			return moment.m03;
	}
	else if (x_order == 1) {
		if (y_order == 0)
			return moment.m10;
		else if (y_order == 1)
			return moment.m11;
		else
			return moment.m12;
	}
	else if (x_order == 2) {
		if (y_order == 0)
			return moment.m20;
		else
			return moment.m21;
	}
	else {
		return moment.m30;
	}
}

double CImageAnalysis::getCentralMoment(cv::Moments moment, int x_order, int y_order)
{
	if (x_order + y_order > 3) {
		return NULL;
	}

	if (x_order == 0) {
		if (y_order == 2)
			return moment.mu02;
		else
			return moment.mu03;
	}
	else if (x_order == 1) {
		if (y_order == 1)
			return moment.mu11;
		else
			return moment.mu12;
	}
	else if (x_order == 2) {
		if (y_order == 0)
			return moment.mu20;
		else
			return moment.mu21;
	}
	else {
		return moment.mu30;
	}
}

double CImageAnalysis::getNormalizedCentralMoment(cv::Moments moment, int x_order, int y_order)
{
	if (x_order + y_order > 3) {
		return NULL;
	}

	if (x_order == 0) {
		if (y_order == 2)
			return moment.nu02;
		else
			return moment.nu03;
	}
	else if (x_order == 1) {
		if (y_order == 1)
			return moment.nu11;
		else
			return moment.nu12;
	}
	else if (x_order == 2) {
		if (y_order == 0)
			return moment.nu20;
		else
			return moment.nu21;
	}
	else {
		return moment.mu30;
	}
}

CImageAnalysis::CImageAnalysis()
{

}

CImageAnalysis::~CImageAnalysis()
{

}

// Pryamid : Up-sampling
cv::Mat CImageAnalysis::GS_pyrUp(cv::Mat src_image)
{
	cv::Mat dst_image = cv::Mat(src_image.size()*2, src_image.type());

	pyrUp(src_image, dst_image, dst_image.size()); // Let's firgure it out!

	return dst_image;
}

// Pryamid : Down-sampling
cv::Mat CImageAnalysis::GS_pyrDown(cv::Mat src_image)
{
	cv::Mat dst_image = cv::Mat(src_image.size() / 2, src_image.type());

	pyrDown(src_image, dst_image, dst_image.size());

	return dst_image;
}

// Pryamid : Segmentation
/*
IplImage *CImageAnalysis::GS_prySegmentation( IplImage *src_image,                               
											  int level, 
											  double threshold1,
											  double threshold2 )
{
	// 초기화
	IplImage *dst_image = cvCreateImage( cvGetSize( src_image ), 
										 IPL_DEPTH_8U, 
										 src_image->nChannels );

	// 
	CvMemStorage* storage = cvCreateMemStorage( 1000 );
	CvSeq* comp1 = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	CvSeq** comp = &comp1;

	// 정방형의 크기어야 함.
	cvPyrSegmentation( src_image, dst_image, storage, comp, level, threshold1, threshold2 ); 	

	return dst_image;
}
*/

// Pryamid : Mean Shifting
cv::Mat CImageAnalysis::GS_pyrMeanShiftFiltering(cv::Mat src_image, double sp, double sr)
{
	CUtil util;

	cv::Mat dst_image = util.GS_createImage(src_image.size(), CV_8U, src_image.channels());

	cv::pyrMeanShiftFiltering(src_image, dst_image, sp, sr);

	return dst_image;
}

// Harris corner detector
cv::Mat CImageAnalysis::GS_cornerHarris(cv::Mat src_image, int block_size, int aperture_size, double k)
{
	CUtil util;
	if (src_image.channels() != 1) return cv::Mat();

	cv::Mat dst_image = cv::Mat(src_image.size(), src_image.type());

	cv::Mat corner_image_32F = util.GS_createImage(src_image.size(), CV_32F, src_image.channels());


	/* Harris corner detector:
	Calculates det(M) - k*(trace(M)^2), where M is 2x2 gradient covariation matrix for each pixel */
	cv::cornerHarris(src_image, corner_image_32F, block_size, aperture_size, k);

	// 32F -> 8U
	cv::convertScaleAbs(corner_image_32F, dst_image, 255, 1);

	// 할당한 메모리 해제
	corner_image_32F.release();
	
	return dst_image;
}

// Calculates constraint image for corner detection
cv::Mat CImageAnalysis::GS_preCornerDetect(cv::Mat src_image, int aperture_size)
{
	CUtil util;

	if (src_image.channels() != 1) return cv::Mat();
	
	cv::Mat dst_image = util.GS_createImage(src_image.size(), CV_8U, src_image.channels());

	cv::Mat corner_image_32F = util.GS_createImage(src_image.size(), CV_32F, src_image.channels());

	/* Calculates constraint image for corner detection
	Dx^2 * Dyy + Dxx * Dy^2 - 2 * Dx * Dy * Dxy.
	Applying threshold to the result gives coordinates of corners */
	cv::preCornerDetect(src_image, corner_image_32F, aperture_size);

	// 32F -> 8U
	cv::convertScaleAbs(corner_image_32F, dst_image, 255, 1);

	// 할당한 메모리 해제
	corner_image_32F.release();

	return dst_image;
}

// 외곽선 탐색 : cvFindContours() + cvDrawContours()
cv::Mat CImageAnalysis::GS_findContours(cv::Mat src_image)
{
	CUtil util;
	// 수행 대상은 명암도 영상임.
	if (src_image.channels() != 1) return cv::Mat();

	// 초기화
	cv::Mat tmp_image = src_image.clone();
	cv::Mat dst_image = util.GS_createImage(src_image.size(), CV_8U, 3);
	cv::cvtColor(src_image, dst_image, cv::COLOR_GRAY2BGR);

	std::vector< std::vector<cv::Point> > contours;


	/* Retrieves outer and optionally inner boundaries of white (non-zero) connected
	components in the black (zero) background */
	cv::findContours(tmp_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	/* Draws contour outlines or filled interiors on the image */

	
	for (size_t i = 0; i < contours.size(); i++)
	{
		
			cv::drawContours(dst_image, contours, (int)i, cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 2, 8);
		
		
	}

	// 할당한 메모리 해제
	tmp_image.release();

	return dst_image;
}

// Hu Moment
double * CImageAnalysis::GS_getHuMoments(cv::Mat src_image, double threshold)
{
	// 수행 대상은 명암도 영상임.
	if (src_image.channels() != 1) return NULL;

	// 이진화
	CPixel cpx;
	double max_value = 255.0;
	int type = cv::THRESH_BINARY;
	cv::Mat binary_image = cpx.GS_threshold(src_image, threshold, max_value, type);

	cv::Moments moments;
	double * hu_moments = new double[7];

	// 모멘트 얻는다. 
	// moments 두 번째 매개변수 false : 무시 
	// moments 두 번째 매개변수 true : 0인 값은 0으로, 나머지 값은 1로 
	moments = cv::moments(binary_image, true);
	cv::HuMoments(moments, hu_moments);

	// 할당한 메모리 반환
	binary_image.release();

	return hu_moments;
}

// 공간, 중앙, 정규화된 중앙 모멘트 얻기(불변 모멘트 제외)
double CImageAnalysis::GS_getAllMoments(cv::Mat src_image, double threshold, int type, int xorder, int yorder)
{
	// 수행 대상은 명암도 영상임.
	if (src_image.channels() != 1) return NULL;

	// 이진화
	CPixel cPixel;
	double max_value = 255.0;
	cv::Mat binary_image = cPixel.GS_threshold(src_image, threshold, max_value, cv::THRESH_BINARY);

	// 
	cv::Moments moments;

	// 모멘트 얻는다. 
	// 두 번째 매개변수가 false 인 경우 공간, 중앙, 정규화된 중앙, 불변 모멘트 얻을 수 있다.
	moments = cv::moments(src_image, false);

	int x_order = 0;
	int y_order = 0;

	double var = -1.0;

	// if x_order = 0, y_order = 0 then u00
	// 단, x_order >=0, x_order + y_order <=3 이어야 한다.

	// 공간 모멘트
	if (type == 0)
	{
		var = this->getSpatialMoment(moments, x_order, y_order);
		
	}
	// 중앙 모멘트
	else if (type == 1)
	{
		var = this->getCentralMoment(moments, x_order, y_order);
	}
	// 정규화된 중앙 모멘트
	else if (type == 2)
	{
		var = this->getNormalizedCentralMoment(moments, x_order, y_order);
	}

	// 할당한 메모리 해제
	binary_image.release();

	return var;
}


// 기본 허프 변환
// http://www-cv.mech.eng.osaka-u.ac.jp/~hamada/openCV/src/sample4-1.cc
cv::Mat CImageAnalysis::GS_basicHoughTransform(cv::Mat src_image)
{
	CUtil util;
	// 수행 대상은 컬러 영상임.
	if (src_image.channels() != 3) return cv::Mat();

	cv::Mat tmp_src_image = src_image.clone();
	cv::Mat dst_image = src_image.clone();

	cv::Mat gray_image = util.GS_createImage(src_image.size(), CV_8U, 1);
	cv::Mat edge_image = util.GS_createImage(src_image.size(), CV_8U, 1);

	cv::cvtColor(tmp_src_image, gray_image, cv::COLOR_BGR2GRAY);
	cv::Canny(gray_image, edge_image, 50, 200, 3);
	cv::cvtColor(edge_image, tmp_src_image, cv::COLOR_GRAY2BGR);
	
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edge_image, lines, 1, CV_PI / 180, 70, 0, 0);

	for (int i = 0; i<lines.size(); i++)
	{
		cv::Vec2f line = lines[i];
		float rho = line[0];
		float theta = line[1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);

		if (fabs(a)<0.001)
		{
			pt1.x = pt2.x = cvRound(rho);
			pt1.y = 0;
			pt2.y = tmp_src_image.rows;
		}
		else if (fabs(b) < 0.001)
		{
			pt1.y = pt2.y = cvRound(rho);
			pt1.x = 0;
			pt2.x = tmp_src_image.cols;
		}
		else
		{
			pt1.x = 0;
			pt1.y = cvRound(rho / b);
			pt2.x = cvRound(rho / a);
			pt2.y = 0;
		}
		cv::line(dst_image, pt1, pt2, cv::Scalar(255, 0, 0), 1, 8);
	}

	return dst_image;
}

cv::Mat CImageAnalysis::GS_basicHoughTransformGray(cv::Mat src_image)
{
	CUtil util;
	// 수행 대상은 명암도 영상임.
	if (src_image.channels() != 1) return cv::Mat();

	cv::Mat tmp_src_image;
	cv::Mat dst_image; cv::cvtColor(src_image, dst_image, cv::COLOR_GRAY2BGR);

	cv::Mat edge_image = util.GS_createImage(src_image.size(), CV_8U, 1);

	cv::Canny(src_image, edge_image, 50, 200, 3);
	//cv::cvtColor(edge_image, tmp_src_image, cv::COLOR_GRAY2BGR);

	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edge_image, lines, 1, CV_PI / 180, 70, 0, 0);

	for (int i = 0; i<lines.size(); i++)
	{
		cv::Vec2f line = lines[i];
		float rho = line[0];
		float theta = line[1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);

		if (fabs(a)<0.001)
		{
			pt1.x = pt2.x = cvRound(rho);
			pt1.y = 0;
			pt2.y = tmp_src_image.rows;
		}
		else if (fabs(b) < 0.001)
		{
			pt1.y = pt2.y = cvRound(rho);
			pt1.x = 0;
			pt2.x = tmp_src_image.cols;
		}
		else
		{
			pt1.x = 0;
			pt1.y = cvRound(rho / b);
			pt2.x = cvRound(rho / a);
			pt2.y = 0;
		}
		cv::line(dst_image, pt1, pt2, cv::Scalar(255, 0, 0), 1, 8);
	}

	return dst_image;
}

// 확률적 허프 변환
// http://www-cv.mech.eng.osaka-u.ac.jp/~hamada/openCV/src/sample4-2.cc
cv::Mat CImageAnalysis::GS_probHoughTransform(cv::Mat src_image)
{
	CUtil util;
	// 수행 대상은 컬러 영상임.
	if (src_image.channels() != 3) return cv::Mat();

	cv::Mat tmp_src_image = src_image.clone();
	cv::Mat dst_image = src_image.clone();


	cv::Mat gray_image = util.GS_createImage(src_image.size(), CV_8U, 1);
	cv::Mat edge_image = util.GS_createImage(src_image.size(), CV_8U, 1);

	cv::cvtColor(tmp_src_image, gray_image, cv::COLOR_BGR2GRAY);
	cv::Canny(gray_image, edge_image, 50, 200, 3);
	cv::cvtColor(edge_image, tmp_src_image, cv::COLOR_GRAY2BGR);

	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(edge_image, lines, 1, CV_PI / 180, 30, 5, 5);

	for (int i = 0; i<lines.size(); i++)
	{
		cv::Point pt1 = cv::Point(lines[i][0], lines[i][1]);
		cv::Point pt2 = cv::Point(lines[i][2], lines[i][3]);
		cv::line(dst_image, pt1, pt2, cv::Scalar(255, 0, 0), 1, 8);
	}

	return dst_image;
}

// 템플릿 매칭(template matching)
// http://www-cv.mech.eng.osaka-u.ac.jp/~hamada/openCV/src/sample15.cc
cv::Mat CImageAnalysis::GS_templateMatching(cv::Mat src_image, cv::Mat template_image)
{
	CUtil util;

	// 수행 대상은 컬러 영상임.
	if (src_image.channels() != 3
		|| template_image.channels() != 3)
	{
		printf("src_image와 template_image의 채널 개수가 일치하지 않습니다!");
		return cv::Mat();
	}

	cv::Mat dst_image = src_image.clone();
	cv::Mat matching_image = util.GS_createImage(cv::Size(src_image.cols - template_image.cols + 1
														, src_image.rows - template_image.rows + 1)
												, CV_32F
												, 1);


	/* Measures similarity between template and overlapped windows in the source image
	and fills the resultant image with the measurements */
	cv::matchTemplate(src_image, template_image, matching_image, cv::TM_SQDIFF);

	double  min_val, max_val;
	cv::Point min_loc, max_loc;

	cv::minMaxLoc(matching_image, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());
	
	cv::rectangle(dst_image
				, min_loc
				, cv::Point(min_loc.x + template_image.cols, min_loc.y + template_image.rows)
				, cv::Scalar(255, 0, 0)
				, 2
				, 8
				, 0);

	return dst_image;
}
