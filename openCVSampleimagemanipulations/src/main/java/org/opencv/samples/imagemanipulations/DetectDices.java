package org.opencv.samples.imagemanipulations;

import android.app.Activity;
import android.os.AsyncTask;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.Console;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Jcama on 29/12/2015.
 */

public class DetectDices extends AsyncTask<Mat, Void, Void> {

    private Mat frame = null;

    public DetectDices() {

    }

    public Void doInBackground(Mat... params) {
        Size sizeRgba = params[0].size();
        Mat rgbaInnerWindow;
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat cl1 = new Mat();
        rgbaInnerWindow = params[0].submat(0, rows, 0, cols);
        Equalizer(params[1], cl1);
        Blur(cl1, rgbaInnerWindow);
        Dilate_Erode(rgbaInnerWindow, cl1);
        Find_Squares(cl1);
        this.frame = cl1.clone();
        return null;
    }

    public Mat GetFrame(){
        return this.frame;
    }

    private void Equalizer(Mat img, Mat cl1) {
        Mat mattemp = new Mat();
        double clipLimit = 1.2;
        Size tileGridSize = new Size(8, 8);
        CLAHE clahe = Imgproc.createCLAHE(clipLimit, tileGridSize);
        clahe.apply(img, mattemp);
        Imgproc.equalizeHist(mattemp, cl1);
    }

    private void Blur(Mat img, Mat blurDst) {
        Size kernel = new Size(5, 5);
        Imgproc.blur(img, blurDst, kernel);
    }

    private void Dilate_Erode(Mat img, Mat erodeDst) {
        Mat dilated_ker = new Mat().ones(1, 1, CvType.CV_32F);
        Mat erorde_ker = new Mat().ones(6, 6, CvType.CV_32F);
        Mat dilateDst = new Mat();
        Imgproc.dilate(img, dilateDst, dilated_ker);
        Imgproc.erode(dilateDst, erodeDst, erorde_ker);
    }

    private void Find_Squares(Mat img) {
        Mat gaussDst = new Mat();
        Size gaussKer = new Size(5, 5);
        Imgproc.GaussianBlur(img, gaussDst, gaussKer, 0);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Mat bin = new Mat();
        List<Mat> listGrays = new ArrayList<Mat>();
        Core.split(gaussDst, listGrays);
        for (Mat gray : listGrays) {
            for (int thrs = 0; thrs <= 255; thrs += 26) {
                if (thrs == 0) {
                    Mat cannyDst = new Mat();
                    Imgproc.Canny(img, cannyDst, 80, 90);
                    int apertureSize = 5;
                    Mat dilated_ker = new Mat().ones(1, 1, CvType.CV_32F);
                    Imgproc.dilate(cannyDst, bin, dilated_ker);
                } else {
                    double retval = Imgproc.threshold(gray, bin, thrs, 255, Imgproc.THRESH_BINARY);
                }

                Mat hierarchy = new Mat();
                Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint cnt : contours) {
                    MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
                    double cnt_len = Imgproc.arcLength(cnt2f, true);
                    Imgproc.approxPolyDP(cnt2f, cnt2f, 0.02 * cnt_len, true);
                    cnt2f.convertTo(cnt, CvType.CV_32S);
                    if ((cnt.size().height == 4) && Imgproc.contourArea(cnt2f) > 1000 && Imgproc.contourArea(cnt2f) < 100000 && Imgproc.isContourConvex(cnt)) {
                        cnt.reshape(-1, 2);
                    }
                }
            }
        }
    }
}

