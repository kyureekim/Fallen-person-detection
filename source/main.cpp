#include<opencv2/opencv.hpp>


#include "Pixel.h"
#include "ImageAnalysis.h"
#include<time.h>
#include <math.h>
#include<vector>

#define T_area 4000 // 임계 면적
using namespace cv;

using namespace std;

void COF(Mat & frame, int *nX, int *nY,int thre)//무게 중심 구하기
{
	int cnt = 0;//무게 카운트
	int mx = 0;//x점
	int my = 0;//y점
	for (int i = 0; i < frame.size().height; i++)//모든 픽셀 check (높이)
	{
		for (int j = 0; j < frame.size().width; j++)//(넓이)
		{
			if (frame.at<uchar>(i, j) >= thre)//(i,j)위치의 1채널(gray-scale or 이진영상) 픽셀에 접근해 밝기 값이 thre보다 크면 
			{//해당 위치의 픽셀의 위치를 저장 mx, my
				//조건에 맞는 픽셀의 전체 개수 cnt 에 저장 
				cnt++;//무게 계산
				mx += j;
				my += i;
			}
		}
	}
	if (cnt == 0)cnt = 1;//없으면 cnt 를 1

	if (cnt > 0)//0보다 크면 나눠서 무게 중심 구함 
	{
		*nX = mx / cnt;
		*nY = my / cnt;
	}
}


void contours_hier(Mat &a,Mat &dst)
{
	Mat src = a;
    if(src.empty()){
        cerr << "image load error" << endl;
        return ;
    }

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    cvtColor(src, dst, COLOR_GRAY2BGR);
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
		if (contourArea(contours[idx]) > T_area) {
			cout << contourArea(contours[idx]) << endl;
			Scalar color(255, 255, 255);// 면적 이상일때만 흰색으로
			drawContours(dst, contours, idx, color, -1, LINE_8, hierarchy);
		}
		else {
			Scalar color(0, 0, 0);// 면적 이상일때만 흰색으로
			drawContours(dst, contours, idx, color, -1, LINE_8, hierarchy);

		}
	}
    imshow("labeling", src);
    imshow("area_labeling", dst);  

}

Mat getHandMask1(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158) {

	//컬러 공간 변환 BGR->YCrCb

	Mat YCrCb;

	cvtColor(image, YCrCb, CV_BGR2YCrCb);



	//각 채널별로 분리

	vector<Mat> planes;


	split(YCrCb, planes);



	//각 채널별로 화소마다 비교

	Mat mask(image.size(), CV_8U, Scalar(0));   //결과 마스크를 저장할 영상

	int nr = image.rows;    //전체 행의 수

	int nc = image.cols;



	for (int i = 0; i < nr; i++) {

		uchar* CrPlane = planes[1].ptr<uchar>(i);   //Cr채널의 i번째 행 주소

		uchar* CbPlane = planes[2].ptr<uchar>(i);   //Cb채널의 i번째 행 주소

		for (int j = 0; j < nc; j++) {

			if ((minCr < CrPlane[j]) && (CrPlane[j] < maxCr) && (minCb < CbPlane[j]) && (CbPlane[j] < maxCb))

				mask.at<uchar>(i, j) = 255;

		}

	}



	return mask;

}
//// =----------------------정지 영상 --------------------------------------=
//int main()
//{
//	CPixel photo;
//	CImageAnalysis CA;
//	int x = 0, y = 0;
//	Mat human2 = imread("6.jpg", IMREAD_REDUCED_COLOR_4);  //이미지 읽어오기
//	Mat human = imread("6.jpg", IMREAD_REDUCED_GRAYSCALE_4);  //이미지 읽어오기
//	Mat tmp = human2.clone();
//	Mat mult = human2.clone();
//	Mat background = imread("background.jpg", IMREAD_REDUCED_GRAYSCALE_4);  //이미지 읽어오기
//	Mat B_result = photo.GS_subtract_image(background, human); //back-human 차영상
//	Mat H_result = photo.GS_subtract_image(human, background); //human-back 차영상
//	Mat sum_of_sub = photo.GS_add_image(B_result, H_result);// 차영상의 합
//	imshow("sub", sum_of_sub);
//	Mat binary = photo.GS_threshold(sum_of_sub, 70, 255, 0);//50 기준으로 이진화 50보다 크면 255 작으면 0
//
//	Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4), cv::Point(1, 1));//SE 생성
//	Mat eroded;//erosion 4*4 SE 로 4번반복 
//	erode(binary, eroded, mask, cv::Point(-1, -1), 4);//(erosion 할 이미지, destination 이미지 , SE, 반복 횟수)
//	Mat dilated;//dilation 4*4 SEfh 4번 반복 
//	dilate(eroded, dilated, mask, cv::Point(-1, -1),12);
//	imshow("erosion", eroded); //출력
//	imshow("dilation", dilated); //출력
//
//	Mat dst;
//	contours_hier(dilated,dst);
//	cvtColor(dst, dst, CV_BGR2GRAY);
//
//	Mat binary1 = photo.GS_threshold(dst, 70, 1, 0);// 머리로 검출된 부분을 1로 채우기 
//	cvtColor(binary1, binary1, CV_GRAY2BGR);// Gray-> color 머리 부분만 (1,1,1)인 영상 
//
//	resize(mult, mult, Size(binary1.cols, binary1.rows), 0, 0, CV_INTER_LINEAR);// 사이즈 변경 
//
//	int nBlue = 0, nGreen = 0, nRed = 0; // 두 영상의 곱 변수 
//	for (int i = 0; i < tmp.size().height; i++) {
//		for (int j = 0; j < tmp.size().width; j++) {
//			nBlue = tmp.at<Vec3b>(i, j)[0] * binary1.at<Vec3b>(i, j)[0];
//			nGreen = tmp.at<Vec3b>(i, j)[1] * binary1.at<Vec3b>(i, j)[1];
//			nRed = tmp.at<Vec3b>(i, j)[2] * binary1.at<Vec3b>(i, j)[2];//rgb 각각의 곱 결과
//			// cramping 고려해 mult에 값 할당 
//			if (nBlue > 255) {
//				mult.at<Vec3b>(i, j)[0] = 255;
//			}
//			else mult.at<Vec3b>(i, j)[0] = nBlue;
//
//			if (nGreen > 255) {
//				mult.at<Vec3b>(i, j)[1] = 255;
//			}
//			else mult.at<Vec3b>(i, j)[1] = nGreen;
//
//			if (nRed > 255) {
//				mult.at<Vec3b>(i, j)[2] = 255;
//			}
//			else mult.at<Vec3b>(i, j)[2] = nRed;
//
//		}
//	}
//	Mat fun=getHandMask1(mult);
//	imshow("mult", mult);
//	imshow("fun", fun);
//	cout << "무게중심: ";
//	COF(dst, &x, &y,255);
//	cout << x << "    " << y << endl;
//	
//
//	waitKey();
//
//	return 0;
//}


//// =----------------------동영상 --------------------------------------=
int main()
{
	int x = 0, y = 0;
	VideoCapture cam(0);//내장 카메라를 자동을 찾아준다
	cam.set(CV_CAP_PROP_FRAME_WIDTH, 640);//영상의 넓이 조절 600
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//영상의 높이 조절 500
	Mat human;  //객체 생성
	clock_t start, end;
	CImageAnalysis CA;//이미지 분석 객체

	int tmp_x = 0, tmp_y = 0;
	int gap_x = 0, gap_y = 0;//속도 차이 
	int tmp_gap_x=0, tmp_gap_y=0;
	int pre_v_x = 0, past_v_x = 0;// 가속도 
	int pre_v_y = 0, past_v_y=0;
	double time = 0;
	Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4), cv::Point(1, 1));//SE 생성
	CPixel photo;
	Mat background = imread("background.jpg", 0);  // 9이미지 읽어오기


	//cam.read(human);
	//imwrite("background.jpg", human);
	//background = imread("background.jpg", 0);
	//imshow("background", background);//실시간 이미지 출력
	while (1)//이미지 실시간으로 읽어오기
	{
		start = clock();
		cam.read(human);//실시간으로 이미지 읽어옴
		Mat tmp = human.clone();
		Mat mult = human.clone();
		imshow("video", human);//실시간 이미지 출력
		cvtColor(human, human, CV_RGB2GRAY);
		resize(background, background, Size(human.cols, human.rows), 0, 0, CV_INTER_LINEAR);//background영상을 video 의 사이즈와 동일하게 변경해 background에 다시 넣어줌
		//+ - 등 연산을 위해 두개의 영상 사이즈를 동일하게 만들어줌 
		imshow("human", human);//실시간 이미지 출력
		if (time > 2) {
			Mat B_result = photo.GS_subtract_image(background, human); //back-human 차영상
			Mat H_result = photo.GS_subtract_image(human, background); //human-back 차영상
			Mat sum_of_sub = photo.GS_add_image(B_result, H_result);// 차영상의 합

			Mat binary = photo.GS_threshold(sum_of_sub, 70, 255, 0);//50 기준으로 이진화 50보다 크면 255 작으면 0
			Mat eroded;//erosion 4*4 SE 로 4번반복 
			erode(binary, eroded, mask, cv::Point(-1, -1), 4);//(erosion 할 이미지, destination 이미지 , SE, 반복 횟수)
			Mat dilated;//dilation 4*4 SEfh 4번 반복 
			dilate(eroded, dilated, mask, cv::Point(-1, -1), 15);
			imshow("erosion", eroded); //출력
			imshow("dilation", dilated); //출력

			Mat dst;
			contours_hier(dilated, dst);
			cvtColor(dst, dst, CV_BGR2GRAY);

			Mat binary1 = photo.GS_threshold(dst, 70, 1, 0);// 머리로 검출된 부분을 1로 채우기 
			cvtColor(binary1, binary1, CV_GRAY2BGR);// Gray-> color 머리 부분만 (1,1,1)인 영상 

			resize(mult, mult, Size(binary1.cols, binary1.rows), 0, 0, CV_INTER_LINEAR);// 사이즈 변경 

			int nBlue = 0, nGreen = 0, nRed = 0; // 두 영상의 곱 변수 
			for (int i = 0; i < tmp.size().height; i++) {
				for (int j = 0; j < tmp.size().width; j++) {
					nBlue = tmp.at<Vec3b>(i, j)[0] * binary1.at<Vec3b>(i, j)[0];
					nGreen = tmp.at<Vec3b>(i, j)[1] * binary1.at<Vec3b>(i, j)[1];
					nRed = tmp.at<Vec3b>(i, j)[2] * binary1.at<Vec3b>(i, j)[2];//rgb 각각의 곱 결과
					// cramping 고려해 mult에 값 할당 
					if (nBlue > 255) {
						mult.at<Vec3b>(i, j)[0] = 255;
					}
					else mult.at<Vec3b>(i, j)[0] = nBlue;

					if (nGreen > 255) {
						mult.at<Vec3b>(i, j)[1] = 255;
					}
					else mult.at<Vec3b>(i, j)[1] = nGreen;

					if (nRed > 255) {
						mult.at<Vec3b>(i, j)[2] = 255;
					}
					else mult.at<Vec3b>(i, j)[2] = nRed;

				}
			}
			Mat fun = getHandMask1(mult);
			imshow("mult", mult);
           	imshow("fun", fun);

			COF(fun, &x, &y,255);//무게 중심
			cout<<"무게중심  x: " << x << "    y: " << y << endl;
			gap_x = abs(tmp_x - x)/2;//현재 속도 x
			gap_y = abs(tmp_y - y)/2;//현재 속도 y
		
			pre_v_x = abs(gap_x - tmp_gap_x) / 2;//가속도 x
			pre_v_y = abs(gap_y - tmp_gap_y) / 2;//가속도 y
		
		
			if (pre_v_x > 5) {
				if (pre_v_y > 5)
				{
					cout << "낙상" << endl;
				}
			}
			tmp_x = x; tmp_y = y; tmp_gap_x = gap_x; tmp_gap_y = gap_y; past_v_x = pre_v_x; // 과거 값 넣어주기 
			time = 0;

		}// time if 문 끝
		end = clock();
		time += (double)(end - start) / CLOCKS_PER_SEC;
	
		if (waitKey(27) == 0) {//esc시 break
			break;
		}
	}



	
	waitKey();
	return 0;
}




