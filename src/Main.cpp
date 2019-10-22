#include    <unistd.h>
#include    <iostream>

#include    <opencv2/calib3d.hpp>
#include    <opencv2/highgui.hpp>
#include    <opencv2/imgproc.hpp>

#include	<jsoncpp/json/value.h>
#include	<jsoncpp/json/reader.h>
#include	<jsoncpp/json/writer.h>
#include 	<fstream>

bool        debug = false, flagPxlFmt = false;
bool 		refFound = false;
int 		tactile = 0;
//	std::system( "xdotool mousemove 300 400" );

int			enableThr = 0, enableThrX = 0, enableDist = 0; 
int         fSettings = 0, conv = 0, scale = 50, sMin = 150, sMax = 255, pxlFormat = 0, nCh = 1, exposure = 156, brightness = 96, contrast = 32, hue = 2000, gain = 0, saturation = 50; // gamma_camera = 0
int 		tmpSMin[] = {sMin, sMin, sMin, sMin, sMin, sMin};		
int 		tmpSMax[] = {sMax, sMax, sMax, sMax, sMax, sMax};
int 		maxConv = 3;

std::vector<cv::Point2f> trapezoidPts;
cv::Mat		pTform;

cv::Size    fs = cv::Size(640, 480);
cv::Mat 	cameraMatrix = cv::Mat::eye(3, 3, CV_64F), distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
cv::Mat 	newCameraMatrix;
cv::ColorConversionCodes codeColor = cv::COLOR_BGR2YUV;

std::string formatParse (int pxlFormat) 
{
	std::string r ; 
	uchar depth = pxlFormat & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (pxlFormat >> CV_CN_SHIFT);

	switch (depth) {
		case CV_8U: 	r = "8U";break;
		case CV_8S: 	r = "8S";break;
		case CV_16U: 	r = "16U";break;
		case CV_16S: 	r = "16S";break;
		case CV_32S: 	r = "32S";break;
		case CV_32F: 	r = "32F";break;
		case CV_64F: 	r = "64F";break;
		default: 		r = "User";break; 
	}	
	r += "C";
	r +=  (chans + '0');

	return r; 
}

void 	callBckFmt(int, void*)
{	
	flagPxlFmt = true;
}	


void 	callBckConvMode(int, void*)
{	
	switch (conv) {
		case 0: 		codeColor = cv::COLOR_BGR2YUV;std::cout << "color space: YUV" << std::endl; break;
		case 1: 		codeColor = cv::COLOR_BGR2HSV;std::cout << "color space: HSV" << std::endl; break;
		case 2: 		codeColor = cv::COLOR_BGR2HLS;std::cout << "color space: HLS" << std::endl; break;
		case 3: 		codeColor = cv::COLOR_BGR2YCrCb;std::cout << "color space: YCrCb" << std::endl; break;
		default: 		codeColor = cv::COLOR_BGR2HSV;std::cout << "color space: HSV" << std::endl;break; 
	}
}	

void displaySettings () {
	//    cv::namedWindow("IN/OUT frame");
	//    cv::namedWindow("IN/OUT frame",cv::WINDOW_AUTOSIZE);
	cv::namedWindow("IN/OUT frame",cv::WINDOW_NORMAL);
	cv::resizeWindow("IN/OUT frame",fs.width, fs.height);
	cv::createTrackbar("color space", "IN/OUT frame", &conv, maxConv, callBckConvMode);
	cv::createTrackbar("freeze settings", "IN/OUT frame", &fSettings, 3);
	cv::createTrackbar("tactile", "IN/OUT frame", &tactile, 1);

	cv::namedWindow("parameters",cv::WINDOW_NORMAL);
	cv::resizeWindow("parameters",200, fs.height);

	cv::createTrackbar("exposure", "parameters", &exposure, 4999);	
	cv::createTrackbar("brightness", "parameters", &brightness, 128);
	cv::createTrackbar("contrast", "parameters", &contrast, 95);
	cv::createTrackbar("saturation", "parameters", &saturation, 128);
	cv::createTrackbar("hue", "parameters", &hue, 4000);
	//    cv::createTrackbar("gamma", "parameters", &gamma_camera, 200);
	cv::createTrackbar("gain", "parameters", &gain, 100);
	cv::createTrackbar("pxlFormat", "parameters", &pxlFormat, 1, callBckFmt);    
	cv::createTrackbar("enable distortion", "parameters", &enableDist, 1);    
	cv::createTrackbar("scale", "parameters", &scale, 100);

	cv::namedWindow("threshold",cv::WINDOW_NORMAL);
	cv::resizeWindow("threshold",200, fs.height);
	cv::createTrackbar("enable threshold", "threshold", &enableThr, 1);
	cv::createTrackbar("thrMin1", "threshold", &tmpSMin[0], 255);
	cv::createTrackbar("thrMax1", "threshold", &tmpSMax[0], 255);
	cv::createTrackbar("thrMin2", "threshold", &tmpSMin[1], 255);
	cv::createTrackbar("thrMax2", "threshold", &tmpSMax[1], 255);
	cv::createTrackbar("thrMin3", "threshold", &tmpSMin[2], 255);
	cv::createTrackbar("thrMax3", "threshold", &tmpSMax[2], 255);
	cv::createTrackbar("thrMin4", "threshold", &tmpSMin[3], 255);
	cv::createTrackbar("thrMax4", "threshold", &tmpSMax[3], 255);
	cv::createTrackbar("thrMin5", "threshold", &tmpSMin[4], 255);
	cv::createTrackbar("thrMax5", "threshold", &tmpSMax[4], 255);
	cv::createTrackbar("thrMin6", "threshold", &tmpSMin[5], 255);
	cv::createTrackbar("thrMax6", "threshold", &tmpSMax[5], 255);
	cv::createTrackbar("enable thr n", "threshold", &enableThrX, 6);
	cv::createTrackbar("minThreshold", "threshold", &sMin, 255);
	cv::createTrackbar("maxThreshold", "threshold", &sMax, 255);	

}

bool 	getROI(cv::Mat img, std::vector<cv::Point2f> &ptvec) {
	cv::Mat rvec,tvec,rot3,vConc;			
	int ww = 9, hh = 6;
	int h = 0;
	double squareSize = 32.3;

	cv::Size bsz(ww,hh);
	ptvec.clear();
	bool ref = cv::findChessboardCorners(img, bsz, ptvec); //,cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
	if (ref)
	{		
		std::vector<cv::Point3f> objPts;
		for( int i = 0; i < hh; ++i )
			for( int j = 0; j < ww; ++j )
				objPts.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));

		bool poseFound = cv::solvePnP(objPts, ptvec, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE  );
		cv::Rodrigues(rvec,rot3);
		cv::vconcat(rot3.t(),tvec.t(),vConc);
		cv::Mat pMat = vConc*newCameraMatrix.t();
		cv::Mat XYZ = (cv::Mat_<double>(4,4) << -1*squareSize,-1*squareSize,-h,1, 9*squareSize,-1*squareSize,-h,1, 9*squareSize,6*squareSize,-h,1, -1*squareSize,6*squareSize,-h,1);
		trapezoidPts.clear();
		for (size_t index = 0; index < XYZ.rows; ++index)
		{
			cv::Mat xyzi = XYZ.row(index)*pMat;
			xyzi = xyzi/xyzi.col(2);
			trapezoidPts.push_back(cv::Point2f(xyzi.at<double>(0,0), xyzi.at<double>(0,1)));
		}
	}
	return ref;
}

int     main(int ac, char **av)
{
	double 		ss = 32.2, rms;
	int 		caseFlag = 0, caseFlagConv = 0;		
	int     	centroidX, centroidY;

	int 		idxChan = 0;	

	cv::Mat 	img, imgConv, imgUndist, imgRes, imgResConv;

	std::vector<cv::Point2f> ptvec;
	std::vector<cv::Vec4i>  hierarchy;
	std::vector<std::vector<cv::Point>> contours; 
	//	cv::Moments mmt;

	cv::Mat		finImg, finImgConv;
	cv::Mat 	R, map1, map2;
	cv::Mat 	channels[3],channelsConv[3];
	std::vector<cv::Mat> channelsMerged, channelsConvMerged;



	cv::FileStorage file;
	if (ac > 1) {
		char    	currentPath[256];
		std::string     calibrationFilePath;

		if (getcwd(currentPath, 256 * sizeof(char)) == NULL)
		{
			std::cerr << "Could not retrieve current path directory." << std::endl;
			return (EXIT_FAILURE);
		}
		calibrationFilePath = std::string(currentPath) + "/" + av[1];

		if (file.open(calibrationFilePath, cv::FileStorage::READ)) 
			std::cout << "\n<config> calibration file in \"" << calibrationFilePath << "\" opened"<<std::endl;
	} else {
		if (file.open("calibration.json", cv::FileStorage::READ)) 
			std::cout << "\n<config> calibration file \"calibration.json\" in current path opened" << std::endl;
	}
	if (!file.isOpened())
	{
		std::cerr << "Could not open calibration file '" << av[1] << "'" << std::endl;
		return (EXIT_FAILURE);
	}

	cameraMatrix.at<double>(0, 0) = file["distortion"]["fx"];
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(0, 2) = file["distortion"]["cx"];

	cameraMatrix.at<double>(1, 0) = 0;
	cameraMatrix.at<double>(1, 1) = file["distortion"]["fy"];
	cameraMatrix.at<double>(1, 2) = file["distortion"]["cy"];

	cameraMatrix.at<double>(2, 0) = 0;
	cameraMatrix.at<double>(2, 1) = 0;
	cameraMatrix.at<double>(2, 2) = 1;

	distCoeffs.at<double>(0) = file["distortion"]["lens_coefficients"]["k1"];
	distCoeffs.at<double>(1) = file["distortion"]["lens_coefficients"]["k2"];
	distCoeffs.at<double>(2) = file["distortion"]["lens_coefficients"]["p1"];
	distCoeffs.at<double>(3) = file["distortion"]["lens_coefficients"]["p2"];
	distCoeffs.at<double>(4) = file["distortion"]["lens_coefficients"]["k3"];
	distCoeffs.at<double>(5) = file["distortion"]["lens_coefficients"]["k4"];
	distCoeffs.at<double>(6) = file["distortion"]["lens_coefficients"]["k5"];
	distCoeffs.at<double>(7) = file["distortion"]["lens_coefficients"]["k6"];

	file.release();    

	//    std::cout << "Camera matrix: " << std::endl << cameraMatrix << std::endl;
	//    std::cout << "Distortion coeffs: \n" << distCoeffs << std::endl;

	R = cv::Mat::eye(3, 3, CV_64F);	
	newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(640, 480), scale / 100.0f, cv::Size(640, 480), 0, true);
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, fs, CV_32FC1, map1, map2);

	cv::VideoCapture camera;
	//	camera.open(ac == 3 ? std::stoi(av[2]) : 0);
	if (ac == 3) 
	{
		if (camera.open(std::stoi(av[2]))) std::cout << "<config> camera dev video" << av[2] << " opened" << std::endl;
	} else {
		if (camera.open("/dev/ELP-USB130W01MT-L21")) std::cout << "<config> camera \"ELP-USB130W01MT-L21\" opened " << std::endl;
	}	

	// windows and trackbar settings 
	displaySettings();

	//command linuxg4v for camera setting : v4l2-ctl -d /dev/video1 --list-ctrls
	camera.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
	//	camera.set(cv::CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
	camera.set(cv::CAP_PROP_FRAME_WIDTH,fs.width);
	camera.set(cv::CAP_PROP_FRAME_HEIGHT,fs.height);

	int fmt = static_cast<int>(camera.get(cv::CAP_PROP_FOURCC));
	char fmtChar[] = {(char)(fmt & 0XFF), (char)((fmt & 0XFF00) >> 8), (char)((fmt & 0XFF0000) >> 16), (char)((fmt & 0XFF000000) >> 24), 0};
	std::cout << "\ndefault piexl foramt: " << fmtChar << std::endl;
	std::cout << "frame size: [" << camera.get(cv::CAP_PROP_FRAME_WIDTH) << "X" << camera.get(cv::CAP_PROP_FRAME_HEIGHT) << "]"<< std::endl;
	std::cout << "pxlFormat: " << formatParse(camera.get(cv::CAP_PROP_FORMAT)) << std::endl;

	// get screen size
	std::cout << "screen resolution: \n" ;
	const char *command="xrandr | grep '*'";
	FILE *fpipe = (FILE*)popen(command,"r");
	char line[256];
	while (fgets(line, sizeof(line), fpipe))
	{	
		std::cout << line;
		//  	  printf("%s", line);
	}
	std::cout << std::endl;	
	pclose(fpipe);

	char subbuff[10];
	memcpy( subbuff, &line[3], 9 );
	subbuff[9] = '\0';
	std::cout << "sub buff: " << subbuff << std::endl;	

	std::vector<char*> v;
	char* chars_array = strtok(subbuff, "x");	
	while(chars_array)
	{
		v.push_back(chars_array);
		chars_array = strtok(NULL, "x");
	}
	std::cout << "width: " << atoi(v[0]) << "\nheight: " << atoi(v[1]) << std::endl;

	while (true)
	{	
		if (fSettings == 0) {
			refFound = false;
			//set camera parameters
			camera.set(cv::CAP_PROP_EXPOSURE, (exposure + 1) / 5000.0);
			camera.set(cv::CAP_PROP_BRIGHTNESS, brightness / 128.0);
			camera.set(cv::CAP_PROP_CONTRAST, contrast / 95.0);
			camera.set(cv::CAP_PROP_SATURATION, saturation / 128.0);
			camera.set(cv::CAP_PROP_HUE, hue / 4000.0);
			camera.set(cv::CAP_PROP_GAIN, gain / 100.0);
			//		camera.set(cv::CAP_PROP_GAMMA, gamma_camera / 200.0);
			if (flagPxlFmt == true) {		
				switch (pxlFormat) {
					case 0:		camera.set(cv::CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
							fmt = static_cast<int>(camera.get(cv::CAP_PROP_FOURCC));
							std::cout << "pixel format: "<< (char)(fmt & 0XFF) << (char)((fmt & 0XFF00) >> 8) << (char)((fmt & 0XFF0000) >> 16) << (char)((fmt & 0XFF000000) >> 24) << std::endl;
							break;
					case 1: 	camera.set(cv::CAP_PROP_FOURCC,CV_FOURCC('Y','U','Y','V'));
							fmt = static_cast<int>(camera.get(cv::CAP_PROP_FOURCC));
							std::cout << "pixel format: "<< (char)(fmt & 0XFF) << (char)((fmt & 0XFF00) >> 8) << (char)((fmt & 0XFF0000) >> 16) << (char)((fmt & 0XFF000000) >> 24) << std::endl;
							break;
							//default:	break;		
				}
			}
			flagPxlFmt = false;
			camera.read(img);	

			if (enableDist == 1) {	
				newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(640, 480), scale / 100.0, cv::Size(640, 480), 0, true);
				cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, fs, CV_32FC1, map1, map2);
				cv::remap(img, img, map1, map2, cv::INTER_LINEAR);
			}

			//convert image 	
			cv::cvtColor(img, imgConv, codeColor);			

			cv::split(img, channels);
			if (enableThrX != 0 && enableThr == 0) {	
				switch (enableThrX) {
					case 1 : 		caseFlag = caseFlag | 0b001; 
								tmpSMin[0] = sMin; 
								tmpSMax[0] = sMax; 
								break;
					case 2 : 		caseFlag = caseFlag | 0b010;  
								tmpSMin[1] = sMin; 
								tmpSMax[1] = sMax; 
								break;
					case 3 : 		caseFlag = caseFlag | 0b100;  
								tmpSMin[2] = sMin; 
								tmpSMax[2] = sMax; 
								break;
				}
				switch (caseFlag) {
					case 0b001 :	cv::threshold(channels[0], channels[0], tmpSMin[0], tmpSMax[0], CV_THRESH_BINARY);break;			
					case 0b010 :	cv::threshold(channels[1], channels[1], tmpSMin[1], tmpSMax[1], CV_THRESH_BINARY);break;
					case 0b100 :	cv::threshold(channels[2], channels[2], tmpSMin[2], tmpSMax[2], CV_THRESH_BINARY);break;
					case 0b011 :	cv::threshold(channels[0], channels[0], tmpSMin[0], tmpSMax[0], CV_THRESH_BINARY);
							cv::threshold(channels[1], channels[1], tmpSMin[1], tmpSMax[1], CV_THRESH_BINARY);break;
					case 0b101 :	cv::threshold(channels[0], channels[0], tmpSMin[0], tmpSMax[0], CV_THRESH_BINARY);
							cv::threshold(channels[2], channels[2], tmpSMin[2], tmpSMax[2], CV_THRESH_BINARY);break;
					case 0b110 :	cv::threshold(channels[1], channels[1], tmpSMin[1], tmpSMax[1], CV_THRESH_BINARY);
							cv::threshold(channels[2], channels[2], tmpSMin[2], tmpSMax[2], CV_THRESH_BINARY);break;
					case 0b111 :	cv::threshold(channels[0], channels[0], tmpSMin[0], tmpSMax[0], CV_THRESH_BINARY);
							cv::threshold(channels[1], channels[1], tmpSMin[1], tmpSMax[1], CV_THRESH_BINARY);
							cv::threshold(channels[2], channels[2], tmpSMin[2], tmpSMax[2], CV_THRESH_BINARY);break;				
				}
				cv::hconcat(channels[0], channels[1], imgRes);
				cv::hconcat(imgRes, channels[2], imgRes);			
			} else {	
				caseFlag = 0; 	
			}

			if (enableThr != 0) {	
				cv::threshold(channels[0], channels[0], tmpSMin[0], tmpSMax[0], CV_THRESH_BINARY);
				cv::threshold(channels[1], channels[1], tmpSMin[1], tmpSMax[1], CV_THRESH_BINARY);
				cv::threshold(channels[2], channels[2], tmpSMin[2], tmpSMax[2], CV_THRESH_BINARY);				
				cv::hconcat(channels[0], channels[1], imgRes);
				cv::hconcat(imgRes, channels[2], imgRes);			
			} 

			if (enableThrX == 0 && enableThr == 0) {		
				cv::split(img, channels);
				//convert image 	
				cv::cvtColor(img, imgConv, codeColor);

				cv::hconcat(channels[0], channels[1], imgRes);
				cv::hconcat(imgRes, channels[2], imgRes);
			}


			cv::split(imgConv, channelsConv);		
			if (enableThrX != 0 && enableThr == 0) {
				switch (enableThrX) {
					case 4 : 		caseFlagConv = caseFlagConv | 0b001; 
								tmpSMin[3] = sMin; 
								tmpSMax[3] = sMax; 
								break;
					case 5 : 		caseFlagConv = caseFlagConv | 0b010;  
								tmpSMin[4] = sMin; 
								tmpSMax[4] = sMax; 
								break;
					case 6 : 		caseFlagConv = caseFlagConv | 0b100;  
								tmpSMin[5] = sMin; 
								tmpSMax[5] = sMax; 
								break;
				}
				switch (caseFlagConv) {
					case 0b001 :	cv::threshold(channelsConv[0], channelsConv[0], tmpSMin[3], tmpSMax[3], CV_THRESH_BINARY);break;			
					case 0b010 :	cv::threshold(channelsConv[1], channelsConv[1], tmpSMin[4], tmpSMax[4], CV_THRESH_BINARY);break;
					case 0b100 :	cv::threshold(channelsConv[2], channelsConv[2], tmpSMin[5], tmpSMax[5], CV_THRESH_BINARY);break;
					case 0b011 :	cv::threshold(channelsConv[0], channelsConv[0], tmpSMin[3], tmpSMax[3], CV_THRESH_BINARY);
							cv::threshold(channelsConv[1], channelsConv[1], tmpSMin[4], tmpSMax[4], CV_THRESH_BINARY);break;
					case 0b101 :	cv::threshold(channelsConv[0], channelsConv[0], tmpSMin[3], tmpSMax[3], CV_THRESH_BINARY);
							cv::threshold(channelsConv[2], channelsConv[2], tmpSMin[5], tmpSMax[5], CV_THRESH_BINARY);break;
					case 0b110 :	cv::threshold(channelsConv[1], channelsConv[1], tmpSMin[4], tmpSMax[4], CV_THRESH_BINARY);
							cv::threshold(channelsConv[2], channelsConv[2], tmpSMin[5], tmpSMax[5], CV_THRESH_BINARY);break;
					case 0b111 :	cv::threshold(channelsConv[0], channelsConv[0], tmpSMin[3], tmpSMax[3], CV_THRESH_BINARY);
							cv::threshold(channelsConv[1], channelsConv[1], tmpSMin[4], tmpSMax[4], CV_THRESH_BINARY);
							cv::threshold(channelsConv[2], channelsConv[2], tmpSMin[5], tmpSMax[5], CV_THRESH_BINARY);break;				
				}
				cv::hconcat(channelsConv[0], channelsConv[1], imgResConv);
				cv::hconcat(imgResConv, channelsConv[2], imgResConv);	
			} else {
				caseFlagConv = 0;
			}

			if (enableThr != 0) {
				cv::threshold(channelsConv[0], channelsConv[0], tmpSMin[3], tmpSMax[3], CV_THRESH_BINARY);
				cv::threshold(channelsConv[1], channelsConv[1], tmpSMin[4], tmpSMax[4], CV_THRESH_BINARY);
				cv::threshold(channelsConv[2], channelsConv[2], tmpSMin[5], tmpSMax[5], CV_THRESH_BINARY);						
				cv::hconcat(channelsConv[0], channelsConv[1], imgResConv);
				cv::hconcat(imgResConv, channelsConv[2], imgResConv);	
			} 
			if (enableThrX == 0 && enableThr == 0) {
				cv::split(imgConv, channelsConv);		
				cv::hconcat(channelsConv[0], channelsConv[1], imgResConv);
				cv::hconcat(imgResConv, channelsConv[2], imgResConv);	
			}

			cv::vconcat(imgRes, imgResConv, imgRes);

			cv::cvtColor(imgRes, imgRes, cv::COLOR_GRAY2BGR);

			channelsMerged.clear();
			channelsMerged.push_back(channels[0]);
			channelsMerged.push_back(channels[1]);
			channelsMerged.push_back(channels[2]);		
			cv::merge(channelsMerged,finImg);		

			channelsConvMerged.clear();
			channelsConvMerged.push_back(channelsConv[0]);
			channelsConvMerged.push_back(channelsConv[1]);
			channelsConvMerged.push_back(channelsConv[2]);		
			cv::merge(channelsConvMerged,finImgConv);

			/*		
					cv::resize(img, img, cv::Size(fs.width*1.5, fs.height*1.5));
					cv::resize(imgConv, imgConv, cv::Size(fs.width*1.5, fs.height*1.5));
					cv::vconcat(img, imgConv, img);		

					cv::resize(finImg, finImg, cv::Size(fs.width*1.5, fs.height*1.5));
					cv::resize(finImgConv, finImgConv, cv::Size(fs.width*1.5, fs.height*1.5));
					cv::vconcat(finImg, finImgConv, finImg);		
					cv::hconcat(img, finImg, finImg);	
					cv::vconcat(finImg, imgRes, imgRes);
					*/
			cv::vconcat(img, imgConv, img);
			cv::vconcat(finImg, finImgConv, finImg);		
			cv::hconcat(img, imgRes, imgRes);		
			cv::hconcat(imgRes, finImg, imgRes);

			cv::imshow("IN/OUT frame", imgRes);			
			cv::waitKey(1);
		} else { 
			camera.read(img);
			idxChan = fSettings - 1;
			cv::split(img, channels);
			cv::threshold(channels[idxChan], channels[idxChan], tmpSMin[idxChan], tmpSMax[idxChan], CV_THRESH_BINARY);	

			cv::remap(channels[idxChan], channels[idxChan], map1, map2, cv::INTER_LINEAR);
			cv::remap(img, imgUndist, map1, map2, cv::INTER_LINEAR);

			if (tactile == 1) {
				if (!refFound) {
					// red channel for getting chessboard corners
					//				refFound = getROI(channels[0], ptvec);
					refFound = getROI(img, ptvec);
					if (refFound) {
						std::cout << "\n<ref> found" << std::endl;					
						std::cout << "<ref> draw chess board corners" << std::endl;

						cv::drawChessboardCorners(img, cv::Size(9,6), ptvec, true);
						cv::imshow("Chessboard", img);
						cv::waitKey(1);

						std::vector<cv::Point2f> rectPts;
						rectPts.push_back(cv::Point2f(0, 0));
						rectPts.push_back(cv::Point2f(fs.width, 0));
						rectPts.push_back(cv::Point2f(fs.width, fs.height));
						rectPts.push_back(cv::Point2f(0, fs.height));
						pTform = cv::getPerspectiveTransform(trapezoidPts, rectPts);
					} else {
						std::cout << "can not find a available reference image " << std::endl; 	
					}

				}	
				else {

					cv::warpPerspective(channels[idxChan],channels[idxChan],pTform,fs);	
					cv::warpPerspective(imgUndist,imgUndist,pTform,fs);																	
					cv::findContours(channels[idxChan], contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
					if (contours.size() > 0) { 
						cv::Moments mmt = cv::moments(contours[0]);
						centroidX = (int) (mmt.m10 / mmt.m00);
						centroidY = (int) (mmt.m01 / mmt.m00);
						//std::cout << "x: " << centroidX << ",y: " << centroidY << std::endl;
						centroidX = (int) (centroidX * 1920.0 / fs.width);
						centroidY = (int) (centroidY * 1080.0 / fs.height);
						std::ostringstream strX,strY;
						strX << centroidX;
						strY << centroidY;
						std::string cmdLinux = "xdotool mousemove " + strX.str() + " " + strY.str();
						std::system(cmdLinux.c_str()); 
						//std::system( "xdotool mousemove 300 400" );
					}

				}	
			}
			cv::cvtColor(channels[idxChan], channels[idxChan], cv::COLOR_GRAY2BGR);
			cv::hconcat(imgUndist, channels[idxChan], imgUndist);
			cv::imshow("IN/OUT frame", imgUndist);	

			cv::waitKey(1);
		}
	}
	return (EXIT_SUCCESS);
}





/*
   std::ofstream fCalib;
   Json::Value event;   
   event["camera"]["device"] = "/dev/video0";
   event["camera"]["frame_width"] = 640;
   event["camera"]["frame_height"] = 480;
   event["distortion"]["cx"] = cameraMatrix.at<double>(0, 2);
   event["distortion"]["cy"] = cameraMatrix.at<double>(1, 2);
   event["distortion"]["fx"] = cameraMatrix.at<double>(0, 0);
   event["distortion"]["fy"] = cameraMatrix.at<double>(1, 1);
   event["distortion"]["free_scaling"] = scale/100.0;
   event["distortion"]["lens_coefficients"]["k1"] = distCoeffs.at<double>(0);
   event["distortion"]["lens_coefficients"]["k2"] = distCoeffs.at<double>(1);
   event["distortion"]["lens_coefficients"]["p1"] = distCoeffs.at<double>(2);
   event["distortion"]["lens_coefficients"]["p2"] = distCoeffs.at<double>(3);
   event["distortion"]["lens_coefficients"]["k3"] = distCoeffs.at<double>(4);
   event["distortion"]["lens_coefficients"]["k4"] = distCoeffs.at<double>(5);
   event["distortion"]["lens_coefficients"]["k5"] = distCoeffs.at<double>(6);
   event["distortion"]["lens_coefficients"]["k6"] = distCoeffs.at<double>(7);
   event["perspective"]["top_left"]["x"] = drawPts[0].x;
   event["perspective"]["top_left"]["y"] = drawPts[0].y;
   event["perspective"]["top_right"]["x"] = drawPts[1].x;
   event["perspective"]["top_right"]["y"] = drawPts[1].y;
   event["perspective"]["bottom_right"]["x"] = drawPts[2].x;
   event["perspective"]["bottom_right"]["y"] = drawPts[2].y;
   event["perspective"]["bottom_left"]["x"] = drawPts[3].x;
   event["perspective"]["bottom_left"]["y"] = drawPts[3].y;
   event["data_sender"]["device"] = "/dev/ttySAC0";
   event["image_processing"][""][""] = "";
   event["tracking"]["min_area"] = 1.0;
   event["tracking"]["max_area"] = 200.0;
   event["tracking"]["centroid_updating_step"] = 1.0;

   fCalib.open("calibration.json");
   Json::StyledWriter styledWriter;
   fCalib << styledWriter.write(event);
   fCalib.close();
   std::cout << "# ... calibration file saved" << "\n" << std::endl;
   */




/*
   cv::Mat img = cv::imread(std::string(currentPath) + "/" + av[1]+ "ref.jpg", cv::IMREAD_COLOR); 	
   cv::Mat rvec,tvec,rot3,vConc;			
   int ww = 8, hh = 5;
   double squareSize = 32.2;

   cv::Size bsz(ww,hh);
   std::vector<cv::Point2f> ptvec;
   bool found = cv::findChessboardCorners(img, bsz, ptvec,cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);		
   std::vector<cv::Point3f> objPts;
   for( int i = 0; i < hh; ++i )
   for( int j = 0; j < ww; ++j )
   objPts.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));

   bool poseFound = cv::solvePnP(objPts, ptvec, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE  );
   cv::Rodrigues(rvec,rot3);
   cv::vconcat(rot3.t(),tvec.t(),vConc);
   pMat = vConc*newCameraMatrix.t();
   XYZ = (cv::Mat_<double>(4,4) << -2*squareSize,-2*squareSize,-h1,1, 14*squareSize,-2*squareSize,-h2,1, 14*squareSize,7*squareSize,-h3,1, -2*squareSize,7*squareSize,-h4,1);

   drawPts.clear();
   for (size_t index = 0; index < XYZ.rows; ++index)
   {
   cv::Mat xyzi = XYZ.row(index)*pMat;
   xyzi = xyzi/xyzi.col(2);
   drawPts.push_back(cv::Point2f(xyzi.at<double>(0,0), xyzi.at<double>(0,1)));
   }		
   cv::line( imgUndist, drawPts[0], drawPts[1], cv::Scalar(110, 220, 0), 4 );
   cv::line( imgUndist, drawPts[1], drawPts[2], cv::Scalar(110, 220, 0), 4 );
   cv::line( imgUndist, drawPts[2], drawPts[3], cv::Scalar(110, 220, 0), 4 );
   cv::line( imgUndist, drawPts[3], drawPts[0], cv::Scalar(110, 220, 0), 4 );
   */
