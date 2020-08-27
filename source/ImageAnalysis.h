// ImageAnalysis.h: interface for the CImageAnalysis class.
//
//////////////////////////////////////////////////////////////////////
//#include<opencv\cv.h>
//#include<opencv\cxcore.h>
//#include<opencv\cvaux.h>
//#include<opencv\highgui.h>
#include <opencv2/opencv.hpp>

#if !defined(AFX_IMAGEANALYSIS_H__02575BF0_7F7D_4A12_8B40_3BCC13F6AE05__INCLUDED_)
#define AFX_IMAGEANALYSIS_H__02575BF0_7F7D_4A12_8B40_3BCC13F6AE05__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CImageAnalysis  
{
private:
	double getSpatialMoment(cv::Moments moment, int x_order, int y_order);
	double getCentralMoment(cv::Moments moment, int x_order, int y_order);
	double getNormalizedCentralMoment(cv::Moments moment, int x_order, int y_order);
public:
	CImageAnalysis();
	virtual ~CImageAnalysis();

	//-----------
	// 피라미드
	//-----------

	// Pryamid : Up-sampling
	cv::Mat GS_pyrUp(cv::Mat src_image);

	// Pryamid : Down-sampling
	cv::Mat GS_pyrDown(cv::Mat src_image);

	// Pryamid : Segmentation
	//cv::Mat GS_pyrSegmentation(cv::Mat src_image, int level = 1, double threshold1 = 10.0, double threshold2 = 60.0);

	// Pryamid : Mean Shifting
	cv::Mat GS_pyrMeanShiftFiltering(cv::Mat src_image, double sp = 20.0, double sr = 30.0);

	//-----------
	// 코너 추출
	//-----------

	// Harris corner detector
	cv::Mat GS_cornerHarris(cv::Mat src_image, int block_size = 3, int aperture_size = 5, double k = 0.04);

	// Calculates constraint image for corner detection
	cv::Mat GS_preCornerDetect(cv::Mat src_image, int aperture_size = 7);

	//-------------
	// 외곽선 탐색
	//-------------

	// 외곽선 탐색 : cv::findContours() + cv::drawContours()
	cv::Mat GS_findContours(cv::Mat src_image);

	//-------------
	// Hu Moment
	//-------------

	// 불변 모멘트 얻기(Hu Moment) 
	double * GS_getHuMoments(cv::Mat src_image, double threshold = 128);

	// 공간, 중앙, 정규화된 중앙 모멘트 얻기(불변 모멘트 제외)
	double GS_getAllMoments(cv::Mat src_image, double threshold = 128, int type = 1, int xorder = 0, int yorder = 0);

	//


	//-------------
	// 허프 변환
	//-------------

	// 기본 허프 변환 (입력: 3채널 영상)
	cv::Mat GS_basicHoughTransform(cv::Mat src_image);

	// 기본 허프 변환 (입력: 1채널 명암도 영상
	cv::Mat GS_basicHoughTransformGray(cv::Mat src_image);

	// 확률적 허프 변환
	cv::Mat GS_probHoughTransform(cv::Mat src_image);

	//-------------
	// 템플릿 매칭
	//-------------
	
	// 템플릿 매칭
	cv::Mat GS_templateMatching(cv::Mat src_image, cv::Mat template_image);
};

#endif // !defined(AFX_IMAGEANALYSIS_H__02575BF0_7F7D_4A12_8B40_3BCC13F6AE05__INCLUDED_)
